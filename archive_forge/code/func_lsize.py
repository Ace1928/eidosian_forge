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
def lsize() -> bool:
    wl = max((len(i) for i in v_items)) + 1
    if not wl:
        wl = 1
    cols = (max_w - 2) // wl or 1
    rows = len(v_items) // cols
    if cols * rows < len(v_items):
        rows += 1
    if rows + 2 >= max_h:
        return False
    shared.rows = rows
    shared.cols = cols
    shared.wl = wl
    return True