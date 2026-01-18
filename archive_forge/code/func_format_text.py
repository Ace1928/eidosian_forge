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
def format_text(text):
    """Lowercases and strips punctuation."""
    text = text.lower().strip()
    if text == 'n/a' or text == '?' or text == 'nan':
        text = EMPTY_TEXT
    text = re.sub('[^\\w\\d]+', ' ', text).replace('_', ' ')
    text = ' '.join(text.split())
    text = text.strip()
    if text:
        return text
    return EMPTY_TEXT