import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging
def get_all_spans(text, max_ngram_length):
    """
    Split a text into all possible ngrams up to 'max_ngram_length'. Split points are white space and punctuation.

    Args:
      text: Text to split.
      max_ngram_length: maximal ngram length.
    Yields:
      Spans, tuples of begin-end index.
    """
    start_indexes = []
    for index, char in enumerate(text):
        if not char.isalnum():
            continue
        if index == 0 or not text[index - 1].isalnum():
            start_indexes.append(index)
        if index + 1 == len(text) or not text[index + 1].isalnum():
            for start_index in start_indexes[-max_ngram_length:]:
                yield (start_index, index + 1)