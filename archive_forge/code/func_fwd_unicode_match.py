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
def fwd_unicode_match(self, text: str) -> Tuple[str, Sequence[str]]:
    """
        Forward match a string starting with a backslash with a list of
        potential Unicode completions.

        Will compute list of Unicode character names on first call and cache it.

        .. deprecated:: 8.6
            You can use :meth:`fwd_unicode_matcher` instead.

        Returns
        -------
        At tuple with:
            - matched text (empty if no matches)
            - list of potential completions, empty tuple  otherwise)
        """
    slashpos = text.rfind('\\')
    if slashpos > -1:
        s = text[slashpos + 1:]
        sup = s.upper()
        candidates = [x for x in self.unicode_names if x.startswith(sup)]
        if candidates:
            return (s, candidates)
        candidates = [x for x in self.unicode_names if sup in x]
        if candidates:
            return (s, candidates)
        splitsup = sup.split(' ')
        candidates = [x for x in self.unicode_names if all((u in x for u in splitsup))]
        if candidates:
            return (s, candidates)
        return ('', ())
    else:
        return ('', ())