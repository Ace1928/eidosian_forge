import curses
import errno
import functools
import math
import os
import platform
import re
import struct
import sys
import time
from typing import (
from ._typing_compat import Literal
import unicodedata
from dataclasses import dataclass
from pygments import format
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from .formatter import BPythonFormatter
from .config import getpreferredencoding, Config
from .keys import cli_key_dispatch as key_dispatch
from . import translations
from .translations import _
from . import repl, inspection
from . import args as bpargs
from .pager import page
from .args import parse as argsparse
def clear_wrapped_lines(self) -> None:
    """Clear the wrapped lines of the current input."""
    height, width = self.scr.getmaxyx()
    max_y = min(self.iy + (self.ix + len(self.s)) // width + 1, height)
    for y in range(self.iy + 1, max_y):
        self.scr.move(y, 0)
        self.scr.clrtoeol()