import collections
import os
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
from ...tokenization_utils import AddedToken, PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging
def subfinder(mylist, pattern):
    matches = []
    indices = []
    for idx, i in enumerate(range(len(mylist))):
        if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
            matches.append(pattern)
            indices.append(idx)
    if matches:
        return (matches[0], indices[0])
    else:
        return (None, 0)