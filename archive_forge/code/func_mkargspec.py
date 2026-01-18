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
def mkargspec(self, topline: inspection.FuncProps, in_arg: Union[str, int, None], down: bool) -> int:
    """This figures out what to do with the argspec and puts it nicely into
        the list window. It returns the number of lines used to display the
        argspec.  It's also kind of messy due to it having to call so many
        addstr() to get the colouring right, but it seems to be pretty
        sturdy."""
    r = 3
    fn = topline.func
    args = topline.argspec.args
    kwargs = topline.argspec.defaults
    _args = topline.argspec.varargs
    _kwargs = topline.argspec.varkwargs
    is_bound_method = topline.is_bound_method
    kwonly = topline.argspec.kwonly
    kwonly_defaults = topline.argspec.kwonly_defaults or dict()
    max_w = int(self.scr.getmaxyx()[1] * 0.6)
    self.list_win.erase()
    self.list_win.resize(3, max_w)
    h, w = self.list_win.getmaxyx()
    self.list_win.addstr('\n  ')
    self.list_win.addstr(fn, get_colpair(self.config, 'name') | curses.A_BOLD)
    self.list_win.addstr(': (', get_colpair(self.config, 'name'))
    maxh = self.scr.getmaxyx()[0]
    if is_bound_method and isinstance(in_arg, int):
        in_arg += 1
    punctuation_colpair = get_colpair(self.config, 'punctuation')
    for k, i in enumerate(args):
        y, x = self.list_win.getyx()
        ln = len(str(i))
        kw = None
        if kwargs and k + 1 > len(args) - len(kwargs):
            kw = repr(kwargs[k - (len(args) - len(kwargs))])
            ln += len(kw) + 1
        if ln + x >= w:
            ty = self.list_win.getbegyx()[0]
            if not down and ty > 0:
                h += 1
                self.list_win.mvwin(ty - 1, 1)
                self.list_win.resize(h, w)
            elif down and h + r < maxh - ty:
                h += 1
                self.list_win.resize(h, w)
            else:
                break
            r += 1
            self.list_win.addstr('\n\t')
        if str(i) == 'self' and k == 0:
            color = get_colpair(self.config, 'name')
        else:
            color = get_colpair(self.config, 'token')
        if k == in_arg or i == in_arg:
            color |= curses.A_BOLD
        self.list_win.addstr(str(i), color)
        if kw is not None:
            self.list_win.addstr('=', punctuation_colpair)
            self.list_win.addstr(kw, get_colpair(self.config, 'token'))
        if k != len(args) - 1:
            self.list_win.addstr(', ', punctuation_colpair)
    if _args:
        if args:
            self.list_win.addstr(', ', punctuation_colpair)
        self.list_win.addstr(f'*{_args}', get_colpair(self.config, 'token'))
    if kwonly:
        if not _args:
            if args:
                self.list_win.addstr(', ', punctuation_colpair)
            self.list_win.addstr('*', punctuation_colpair)
        marker = object()
        for arg in kwonly:
            self.list_win.addstr(', ', punctuation_colpair)
            color = get_colpair(self.config, 'token')
            if arg == in_arg:
                color |= curses.A_BOLD
            self.list_win.addstr(arg, color)
            default = kwonly_defaults.get(arg, marker)
            if default is not marker:
                self.list_win.addstr('=', punctuation_colpair)
                self.list_win.addstr(repr(default), get_colpair(self.config, 'token'))
    if _kwargs:
        if args or _args or kwonly:
            self.list_win.addstr(', ', punctuation_colpair)
        self.list_win.addstr(f'**{_kwargs}', get_colpair(self.config, 'token'))
    self.list_win.addstr(')', punctuation_colpair)
    return r