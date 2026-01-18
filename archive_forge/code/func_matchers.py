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
@property
def matchers(self) -> List[Matcher]:
    """All active matcher routines for completion"""
    if self.dict_keys_only:
        return [self.dict_key_matcher]
    if self.use_jedi:
        return [*self.custom_matchers, *self._backslash_combining_matchers, *self.magic_arg_matchers, self.custom_completer_matcher, self.magic_matcher, self._jedi_matcher, self.dict_key_matcher, self.file_matcher]
    else:
        return [*self.custom_matchers, *self._backslash_combining_matchers, *self.magic_arg_matchers, self.custom_completer_matcher, self.dict_key_matcher, self.magic_matcher, self.python_matches, self.file_matcher, self.python_func_kw_matcher]