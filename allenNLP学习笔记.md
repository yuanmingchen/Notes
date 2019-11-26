### 1.BucketIterator

Instance：表示一条数据，包含文本、标签等各个字段，每个字段由一个Field表示，针对不同类型的字段，allenNLP内置了多种类型的Field。最常用的莫过于TextField和LabelField了。

顾名思义，TextField封装一个文本字段，构造参数有两个：

- tokens：该文本分词后的token数组，是一个List[Token]，Token也是allenNLP的内置类，用来表示一个文本单元（一个词或一个字）。

- token_indexers：单词索引器，以SingleIdTokenIndexer，其构造参数为:

- ```python
  def __init__(self,
                   namespace: str = 'tokens',
                   lowercase_tokens: bool = False,
                   start_tokens: List[str] = None,
                   end_tokens: List[str] = None) -> None:
  ```

  - Namespace：命名空间。命名空间表示该索引器索引的Token序列的种类，默认为tokens，如果输入的文本只有一种，那么用默认即可。但是如果输入多种文本，比如QA问题，输入一个问题和一个答案，并且向分别给问题和答案使用不同的词向量，那么问题和答案就是两个不同的文本，需要使用不同的词汇表，也就需要不同的两个索引器，这两个索引器就需要使用不同的命名空间表示，说白了，所谓的命名空间，就是当前这个类的实例化对象的名字，通过它可以区分同一类的不同对象，虽然是同一个类，但是由于某些参数不一样，所以他们也需要有所区分。allenNLP中有很多类的构造参数都有命名空间，都是类似的功能。
  - lowercase_tokens：是否需要对所有的token小写，对英文有用。若设为True，则转为id和统计词个数的时候都会先将token小写。当然这并不会影响Token对象原始文本的内容，但是需要传入的词汇表也是全小写的单词。
  - start_tokens：需要在每个单词序列最前面都加入的token，比如bert中的\<CLS\>标签，或者是序列的开始标志
  - end_tokens：需要在每个单词序列最后都加入的token，比如bert中的\<SEP\>标签，或者是序列的结束标志

- 具有以下几个重要功能：

  - `tokens_to_indices(self,tokens: List[Token], vocabulary: Vocabulary, index_name: str)`，将token根据一个词汇表转换成id，支持在前后插入其他token，非常适合bert等，在前面加入CLS等标签。转换成的id放在Token类的text_id属性里面（不是idx，idx表示token的位置，text_id表示token的文本在词汇Vocabulary中的id）。

  - `def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]])`:统计某个token的个数，通过token_id标识token，结果存储在传入的参数counter内，counter为一个字典Dict类型，核心代码counter\[self.namespace\][text] += 1，可以看到是将当前命名空间的当前文本的次数加一。

  - pad_token_sequence: 用于将token序列padding到一定长度。支持多个序列，每个序列有一个字符串表示，类似于命名空间的功能。
  
  - ```python
    def pad_token_sequence(self,
                               tokens: Dict[str, List[int]],
                               desired_num_tokens: Dict[str, int],
                             padding_lengths: Dict[str, int]) -> Dict[str, List[int]]: 
    ```
  
- Trainer把BatchIterator当做方法调用，所以调用了父类DataIterator的\__call__方法，这个方法会调用Batch类的as_tensor_dict方法，该方法计算出实例的每个字段应该padding称多长之后，又会调用Instance的as_tensor方法，该类又会调用每个field的as_tensor方法，该方法又会调用TokenIndexer类的pad_token_sequence方法，最终实现padding。

