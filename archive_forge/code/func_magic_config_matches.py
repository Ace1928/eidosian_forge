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
def magic_config_matches(self, text: str) -> List[str]:
    """Match class names and attributes for %config magic.

        .. deprecated:: 8.6
            You can use :meth:`magic_config_matcher` instead.
        """
    texts = text.strip().split()
    if len(texts) > 0 and (texts[0] == 'config' or texts[0] == '%config'):
        classes = sorted(set([c for c in self.shell.configurables if c.__class__.class_traits(config=True)]), key=lambda x: x.__class__.__name__)
        classnames = [c.__class__.__name__ for c in classes]
        if len(texts) == 1:
            return classnames
        classname_texts = texts[1].split('.')
        classname = classname_texts[0]
        classname_matches = [c for c in classnames if c.startswith(classname)]
        if texts[1].find('.') < 0:
            return classname_matches
        elif len(classname_matches) == 1 and classname_matches[0] == classname:
            cls = classes[classnames.index(classname)].__class__
            help = cls.class_get_help()
            help = re.sub(re.compile('^--', re.MULTILINE), '', help)
            return [attr.split('=')[0] for attr in help.strip().splitlines() if attr.startswith(texts[1])]
    return []