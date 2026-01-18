import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
def find_next_punctuation(text: str, start_idx=0):
    """
    Find the index of the next punctuation mark.

    Args:
        text (`str`):
            String to examine
        start_idx (`int`, *optional*)
            Index where to start
    """
    for i in range(start_idx, len(text)):
        if text[i] in ['.', '?', '!', '\n']:
            return i
    return None