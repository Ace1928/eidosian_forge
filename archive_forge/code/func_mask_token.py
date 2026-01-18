import json
from typing import List, Optional, Tuple
from tokenizers import pre_tokenizers, processors
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_bart import BartTokenizer
@mask_token.setter
def mask_token(self, value):
    """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on Bart.
        """
    value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
    self._mask_token = value