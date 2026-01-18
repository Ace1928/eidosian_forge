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
def rectify_completions(text: str, completions: _IC, *, _debug: bool=False) -> _IC:
    """
    Rectify a set of completions to all have the same ``start`` and ``end``

    .. warning::

        Unstable

        This function is unstable, API may change without warning.
        It will also raise unless use in proper context manager.

    Parameters
    ----------
    text : str
        text that should be completed.
    completions : Iterator[Completion]
        iterator over the completions to rectify
    _debug : bool
        Log failed completion

    Notes
    -----
    :any:`jedi.api.classes.Completion` s returned by Jedi may not have the same start and end, though
    the Jupyter Protocol requires them to behave like so. This will readjust
    the completion to have the same ``start`` and ``end`` by padding both
    extremities with surrounding text.

    During stabilisation should support a ``_debug`` option to log which
    completion are return by the IPython completer and not found in Jedi in
    order to make upstream bug report.
    """
    warnings.warn('`rectify_completions` is a provisional API (as of IPython 6.0). It may change without warnings. Use in corresponding context manager.', category=ProvisionalCompleterWarning, stacklevel=2)
    completions = list(completions)
    if not completions:
        return
    starts = (c.start for c in completions)
    ends = (c.end for c in completions)
    new_start = min(starts)
    new_end = max(ends)
    seen_jedi = set()
    seen_python_matches = set()
    for c in completions:
        new_text = text[new_start:c.start] + c.text + text[c.end:new_end]
        if c._origin == 'jedi':
            seen_jedi.add(new_text)
        elif c._origin == 'IPCompleter.python_matches':
            seen_python_matches.add(new_text)
        yield Completion(new_start, new_end, new_text, type=c.type, _origin=c._origin, signature=c.signature)
    diff = seen_python_matches.difference(seen_jedi)
    if diff and _debug:
        print('IPython.python matches have extras:', diff)