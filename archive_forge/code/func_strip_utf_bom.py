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
def strip_utf_bom(self) -> None:
    """Strip the UTF bom from the lines of the file."""
    if not self.lines:
        return
    if self.lines[0][:1] == '\ufeff':
        self.lines[0] = self.lines[0][1:]
    elif self.lines[0][:3] == 'ï»¿':
        self.lines[0] = self.lines[0][3:]