from __future__ import annotations
import builtins as builtin_mod
import enum
import glob
import inspect
import itertools
import keyword
import os
import re
import string
import sys
import tokenize
import time
import unicodedata
import uuid
import warnings
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace
from typing import (
from IPython.core.guarded_eval import guarded_eval, EvaluationContext
from IPython.core.error import TryNext
from IPython.core.inputtransformer2 import ESC_MAGIC
from IPython.core.latex_symbols import latex_symbols, reverse_latex_symbol
from IPython.core.oinspect import InspectColors
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import generics
from IPython.utils.decorators import sphinx_options
from IPython.utils.dir2 import dir2, get_real_method
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.path import ensure_dir_exists
from IPython.utils.process import arg_split
from traitlets import (
from traitlets.config.configurable import Configurable
import __main__
def dict_key_matches(self, text: str) -> List[str]:
    """Match string keys in a dictionary, after e.g. ``foo[``.

        .. deprecated:: 8.6
            You can use :meth:`dict_key_matcher` instead.
        """
    if self.text_until_cursor.strip().endswith(']'):
        return []
    match = DICT_MATCHER_REGEX.search(self.text_until_cursor)
    if match is None:
        return []
    expr, prior_tuple_keys, key_prefix = match.groups()
    obj = self._evaluate_expr(expr)
    if obj is not_found:
        return []
    keys = self._get_keys(obj)
    if not keys:
        return keys
    tuple_prefix = guarded_eval(prior_tuple_keys, EvaluationContext(globals=self.global_namespace, locals=self.namespace, evaluation=self.evaluation, in_subscript=True))
    closing_quote, token_offset, matches = match_dict_keys(keys, key_prefix, self.splitter.delims, extra_prefix=tuple_prefix)
    if not matches:
        return []
    text_start = len(self.text_until_cursor) - len(text)
    if key_prefix:
        key_start = match.start(3)
        completion_start = key_start + token_offset
    else:
        key_start = completion_start = match.end()
    if text_start > key_start:
        leading = ''
    else:
        leading = text[text_start:completion_start]
    can_close_quote = False
    can_close_bracket = False
    continuation = self.line_buffer[len(self.text_until_cursor):].strip()
    if continuation.startswith(closing_quote):
        continuation = continuation[len(closing_quote):]
    else:
        can_close_quote = True
    continuation = continuation.strip()
    has_known_tuple_handling = isinstance(obj, dict)
    can_close_bracket = not continuation.startswith(']') and self.auto_close_dict_keys
    can_close_tuple_item = not continuation.startswith(',') and has_known_tuple_handling and self.auto_close_dict_keys
    can_close_quote = can_close_quote and self.auto_close_dict_keys
    if not can_close_quote and (not can_close_bracket) and closing_quote:
        return [leading + k for k in matches]
    results = []
    end_of_tuple_or_item = _DictKeyState.END_OF_TUPLE | _DictKeyState.END_OF_ITEM
    for k, state_flag in matches.items():
        result = leading + k
        if can_close_quote and closing_quote:
            result += closing_quote
        if state_flag == end_of_tuple_or_item:
            pass
        if state_flag in end_of_tuple_or_item and can_close_bracket:
            result += ']'
        if state_flag == _DictKeyState.IN_TUPLE and can_close_tuple_item:
            result += ', '
        results.append(result)
    return results