import io
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def convert_ids_to_string(self, ids):
    """
        Converts a sequence of tokens (strings for sub-words) in a single string.
        """
    tokens = self.convert_ids_to_tokens(ids)
    out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
    return out_string