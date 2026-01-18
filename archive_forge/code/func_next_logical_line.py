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
def next_logical_line(self) -> None:
    """Record the previous logical line.

        This also resets the tokens list and the blank_lines count.
        """
    if self.logical_line:
        self.previous_indent_level = self.indent_level
        self.previous_logical = self.logical_line
        if not self.indent_level:
            self.previous_unindented_logical_line = self.logical_line
    self.blank_lines = 0
    self.tokens = []