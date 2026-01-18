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
@functools.cached_property
def file_tokens(self) -> list[tokenize.TokenInfo]:
    """Return the complete set of tokens for a file."""
    line_iter = iter(self.lines)
    return list(tokenize.generate_tokens(lambda: next(line_iter)))