import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
def next_indentation(self) -> int:
    """Return the indentation of the next line based on the current
        input buffer."""
    if self.buffer:
        indentation = next_indentation(self.buffer[-1], self.config.tab_length)
        if indentation and self.config.dedent_after > 0:

            def line_is_empty(line):
                return not line.strip()
            empty_lines = takewhile(line_is_empty, reversed(self.buffer))
            if sum((1 for _ in empty_lines)) >= self.config.dedent_after:
                indentation -= 1
    else:
        indentation = 0
    return indentation