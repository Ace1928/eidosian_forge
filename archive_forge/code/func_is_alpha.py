import io
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def is_alpha(self, char):
    """
        is_alpha
        """
    if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
        return True
    return False