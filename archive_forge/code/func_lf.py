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
def lf(self) -> None:
    """Process a linefeed character; it only needs to check the
        cursor position and move appropriately so it doesn't clear
        the current line after the cursor."""
    if self.cpos:
        for _ in range(self.cpos):
            self.mvc(-1)
    self.print_line(self.s, newline=True)
    self.echo('\n')