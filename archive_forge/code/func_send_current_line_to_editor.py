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
def send_current_line_to_editor(self) -> str:
    lines = self.send_to_external_editor(self.s).split('\n')
    self.s = ''
    self.print_line(self.s)
    while lines and (not lines[-1]):
        lines.pop()
    if not lines:
        return ''
    self.f_string = ''
    self.cpos = -1
    self.iy, self.ix = self.scr.getyx()
    self.evaluating = True
    for line in lines:
        self.stdout_hist += line + '\n'
        self.history.append(line)
        self.print_line(line)
        self.screen_hist[-1] += self.f_string
        self.scr.addstr('\n')
        self.more = self.push(line)
        self.prompt(self.more)
        self.iy, self.ix = self.scr.getyx()
    self.evaluating = False
    self.cpos = 0
    indent = repl.next_indentation(self.s, self.config.tab_length)
    self.s = ''
    self.scr.refresh()
    if self.buffer:
        for _ in range(indent):
            self.tab()
    self.print_line(self.s)
    self.scr.redrawwin()
    return ''