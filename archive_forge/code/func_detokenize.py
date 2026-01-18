from abc import ABC
from abc import abstractmethod
from tokenizers import Tokenizer
from rwkv_tokenizer import RWKV_TOKENIZER, TRIE_TOKENIZER
from typing import List, Union
def detokenize(self, token_ids):
    return self.tokenizer.decode(token_ids)