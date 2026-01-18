import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from ..models.bert import BertTokenizer, BertTokenizerFast
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """
    if not hasattr(tokenizer, 'deprecation_warnings'):
        return tokenizer.pad(*pad_args, **pad_kwargs)
    warning_state = tokenizer.deprecation_warnings.get('Asking-to-pad-a-fast-tokenizer', False)
    tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True
    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = warning_state
    return padded