import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy
from ....utils import logging
def convert_tokens_to_string(self, tokens):
    """Converts a sequence of tokens (string) in a single string."""
    text = ''.join(tokens)
    text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
    return text