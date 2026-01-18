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
def bs_word(self) -> str:
    self.rl_history.reset()
    pos = len(self.s) - self.cpos - 1
    deleted = []
    while pos >= 0 and self.s[pos] == ' ':
        deleted.append(self.s[pos])
        pos -= self.bs()
    while pos >= 0 and self.s[pos] != ' ':
        deleted.append(self.s[pos])
        pos -= self.bs()
    return ''.join(reversed(deleted))