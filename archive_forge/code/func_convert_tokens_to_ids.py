import bisect
import itertools
import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, overload
from .tokenization_utils_base import (
from .utils import PaddingStrategy, TensorType, add_end_docstrings, logging
def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
    """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
    if tokens is None:
        return None
    if isinstance(tokens, str):
        return self._convert_token_to_id_with_added_voc(tokens)
    ids = []
    for token in tokens:
        ids.append(self._convert_token_to_id_with_added_voc(token))
    return ids