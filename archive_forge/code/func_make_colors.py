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
def make_colors(config: Config) -> Dict[str, int]:
    """Init all the colours in curses and bang them into a dictionary"""
    c = {'k': 0, 'r': 1, 'g': 2, 'y': 3, 'b': 4, 'm': 5, 'c': 6, 'w': 7, 'd': -1}
    if platform.system() == 'Windows':
        c = dict(list(c.items()) + [('K', 8), ('R', 9), ('G', 10), ('Y', 11), ('B', 12), ('M', 13), ('C', 14), ('W', 15)])
    for i in range(63):
        if i > 7:
            j = i // 8
        else:
            j = c[config.color_scheme['background']]
        curses.init_pair(i + 1, i % 8, j)
    return c