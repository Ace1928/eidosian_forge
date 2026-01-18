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
def curses_wrapper(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Like curses.wrapper(), but reuses stdscr when called again."""
    global stdscr
    if stdscr is None:
        stdscr = curses.initscr()
    try:
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(1)
        try:
            curses.start_color()
        except curses.error:
            pass
        return func(stdscr, *args, **kwargs)
    finally:
        stdscr.keypad(0)
        curses.echo()
        curses.nocbreak()
        curses.endwin()