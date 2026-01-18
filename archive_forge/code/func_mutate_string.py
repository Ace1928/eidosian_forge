from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
def mutate_string(text: str) -> str:
    """Replace contents with 'xxx' to prevent syntax matching.

    >>> mutate_string('"abc"')
    '"xxx"'
    >>> mutate_string("'''abc'''")
    "'''xxx'''"
    >>> mutate_string("r'abc'")
    "r'xxx'"
    """
    start = text.index(text[-1]) + 1
    end = len(text) - 1
    if text[-3:] in ('"""', "'''"):
        start += 2
        end -= 2
    return text[:start] + 'x' * (end - start) + text[end:]