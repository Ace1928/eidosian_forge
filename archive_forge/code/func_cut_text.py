import bisect
import itertools
import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, overload
from .tokenization_utils_base import (
from .utils import PaddingStrategy, TensorType, add_end_docstrings, logging
def cut_text(self, text, offsets):
    offsets.append(len(text))
    tokens = []
    start = 0
    for end in offsets:
        if start > end:
            logger.error('There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway.')
            continue
        elif start == end:
            continue
        tokens.append(text[start:end])
        start = end
    return tokens