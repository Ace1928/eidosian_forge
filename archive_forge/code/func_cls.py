from abc import ABC
from abc import abstractmethod
from tokenizers import Tokenizer
from rwkv_tokenizer import RWKV_TOKENIZER, TRIE_TOKENIZER
from typing import List, Union
@property
def cls(self):
    raise NotImplementedError('CLS is not provided for {} tokenizer'.format(self.name))