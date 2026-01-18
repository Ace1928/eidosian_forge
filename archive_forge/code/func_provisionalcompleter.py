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
@skip_doctest
@contextmanager
def provisionalcompleter(action='ignore'):
    """
    This context manager has to be used in any place where unstable completer
    behavior and API may be called.

    >>> with provisionalcompleter():
    ...     completer.do_experimental_things() # works

    >>> completer.do_experimental_things() # raises.

    .. note::

        Unstable

        By using this context manager you agree that the API in use may change
        without warning, and that you won't complain if they do so.

        You also understand that, if the API is not to your liking, you should report
        a bug to explain your use case upstream.

        We'll be happy to get your feedback, feature requests, and improvements on
        any of the unstable APIs!
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action, category=ProvisionalCompleterWarning)
        yield