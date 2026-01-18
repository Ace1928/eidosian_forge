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
def read_lines_from_filename(self) -> list[str]:
    """Read the lines for a file."""
    try:
        with tokenize.open(self.filename) as fd:
            return fd.readlines()
    except (SyntaxError, UnicodeError):
        with open(self.filename, encoding='latin-1') as fd:
            return fd.readlines()