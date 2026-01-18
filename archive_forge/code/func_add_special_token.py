import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as sp
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
def add_special_token(self, token):
    if token not in self.special_tokens:
        self.special_tokens.append(token)
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) - 1
            self.ids_to_tokens.append(token)
    return self.id(token)