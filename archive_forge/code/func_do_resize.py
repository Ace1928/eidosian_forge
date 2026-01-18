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
def do_resize(caller: CLIRepl) -> None:
    """This needs to hack around readline and curses not playing
    nicely together. See also gethw() above."""
    global DO_RESIZE
    h, w = gethw()
    if not h:
        return
    curses.endwin()
    os.environ['LINES'] = str(h)
    os.environ['COLUMNS'] = str(w)
    curses.doupdate()
    DO_RESIZE = False
    try:
        caller.resize()
    except curses.error:
        pass