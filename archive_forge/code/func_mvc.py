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
def mvc(self, i: int, refresh: bool=True) -> bool:
    """This method moves the cursor relatively from the current
        position, where:
            0 == (right) end of current line
            length of current line len(self.s) == beginning of current line
        and:
            current cursor position + i
            for positive values of i the cursor will move towards the beginning
            of the line, negative values the opposite."""
    y, x = self.scr.getyx()
    if self.cpos == 0 and i < 0:
        return False
    if x == self.ix and y == self.iy and (i >= 1):
        return False
    h, w = gethw()
    if x - i < 0:
        y -= 1
        x = w
    if x - i >= w:
        y += 1
        x = 0 + i
    self.cpos += i
    self.scr.move(y, x - i)
    if refresh:
        self.scr.refresh()
    return True