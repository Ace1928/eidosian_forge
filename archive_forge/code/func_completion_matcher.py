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
def completion_matcher(*, priority: Optional[float]=None, identifier: Optional[str]=None, api_version: int=1):
    """Adds attributes describing the matcher.

    Parameters
    ----------
    priority : Optional[float]
        The priority of the matcher, determines the order of execution of matchers.
        Higher priority means that the matcher will be executed first. Defaults to 0.
    identifier : Optional[str]
        identifier of the matcher allowing users to modify the behaviour via traitlets,
        and also used to for debugging (will be passed as ``origin`` with the completions).

        Defaults to matcher function's ``__qualname__`` (for example,
        ``IPCompleter.file_matcher`` for the built-in matched defined
        as a ``file_matcher`` method of the ``IPCompleter`` class).
    api_version: Optional[int]
        version of the Matcher API used by this matcher.
        Currently supported values are 1 and 2.
        Defaults to 1.
    """

    def wrapper(func: Matcher):
        func.matcher_priority = priority or 0
        func.matcher_identifier = identifier or func.__qualname__
        func.matcher_api_version = api_version
        if TYPE_CHECKING:
            if api_version == 1:
                func = cast(MatcherAPIv1, func)
            elif api_version == 2:
                func = cast(MatcherAPIv2, func)
        return func
    return wrapper