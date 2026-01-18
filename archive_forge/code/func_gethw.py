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
def gethw() -> Tuple[int, int]:
    """I found this code on a usenet post, and snipped out the bit I needed,
    so thanks to whoever wrote that, sorry I forgot your name, I'm sure you're
    a great guy.

    It's unfortunately necessary (unless someone has any better ideas) in order
    to allow curses and readline to work together. I looked at the code for
    libreadline and noticed this comment:

        /* This is the stuff that is hard for me.  I never seem to write good
           display routines in C.  Let's see how I do this time. */

    So I'm not going to ask any questions.

    """
    if platform.system() != 'Windows':
        h, w = struct.unpack('hhhh', fcntl.ioctl(sys.__stdout__, termios.TIOCGWINSZ, '\x00' * 8))[0:2]
    else:
        from ctypes import windll, create_string_buffer
        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx, maxy = struct.unpack('hhhhHhhhhhh', csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
        elif stdscr:
            sizex, sizey = stdscr.getmaxyx()
        h, w = (sizey, sizex)
    return (h, w)