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
def settext(self, s: str, c: Optional[int]=None, p: bool=False) -> None:
    """Set the text on the status bar to a new permanent value; this is the
        value that will be set after a prompt or message. c is the optional
        curses colour pair to use (if not specified the last specified colour
        pair will be used).  p is True if the cursor is expected to stay in the
        status window (e.g. when prompting)."""
    self.win.erase()
    if len(s) >= self.w:
        s = s[:self.w - 1]
    self.s = s
    if c:
        self.c = c
    if s:
        if self.c:
            self.win.addstr(s, self.c)
        else:
            self.win.addstr(s)
    if not p:
        self.win.noutrefresh()
        self.pwin.refresh()
    else:
        self.win.refresh()