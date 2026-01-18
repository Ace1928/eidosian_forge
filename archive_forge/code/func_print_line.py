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
def print_line(self, s: Optional[str], clr: bool=False, newline: bool=False) -> None:
    """Chuck a line of text through the highlighter, move the cursor
        to the beginning of the line and output it to the screen."""
    if not s:
        clr = True
    if self.highlighted_paren is not None:
        lineno = self.highlighted_paren[0]
        tokens = self.highlighted_paren[1]
        self.reprint_line(lineno, tokens)
        self.highlighted_paren = None
    if self.config.syntax and (not self.paste_mode or newline):
        o = format(self.tokenize(s, newline), self.formatter)
    else:
        o = s
    self.f_string = o
    self.scr.move(self.iy, self.ix)
    if clr:
        self.scr.clrtoeol()
    if clr and (not s):
        self.scr.refresh()
    if o:
        for t in o.split('\x04'):
            self.echo(t.rstrip('\n'))
    if self.cpos:
        t = self.cpos
        for _ in range(self.cpos):
            self.mvc(1)
        self.cpos = t