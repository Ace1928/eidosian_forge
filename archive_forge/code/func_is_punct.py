import io
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def is_punct(self, char):
    """
        is_punct
        """
    if char in ',;:.?!~，；：。？！《》【】':
        return True
    return False