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
def show_list(self, items: List[str], arg_pos: Union[str, int, None], topline: Optional[inspection.FuncProps]=None, formatter: Optional[Callable]=None, current_item: Optional[str]=None) -> None:
    v_items: Collection
    shared = ShowListState()
    y, x = self.scr.getyx()
    h, w = self.scr.getmaxyx()
    down = y < h // 2
    if down:
        max_h = h - y
    else:
        max_h = y + 1
    max_w = int(w * self.config.cli_suggestion_width)
    self.list_win.erase()
    if items and formatter:
        items = [formatter(x) for x in items]
        if current_item is not None:
            current_item = formatter(current_item)
    if topline:
        height_offset = self.mkargspec(topline, arg_pos, down) + 1
    else:
        height_offset = 0

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
    if items:
        v_items = [items[0][:max_w - 3]]
        lsize()
    else:
        v_items = []
    for i in items[1:]:
        v_items.append(i[:max_w - 3])
        if not lsize():
            del v_items[-1]
            v_items[-1] = '...'
            break
    rows = shared.rows
    if rows + height_offset < max_h:
        rows += height_offset
        display_rows = rows
    else:
        display_rows = rows + height_offset
    cols = shared.cols
    wl = shared.wl
    if topline and (not v_items):
        w = max_w
    elif wl + 3 > max_w:
        w = max_w
    else:
        t = (cols + 1) * wl + 3
        if t > max_w:
            t = max_w
        w = t
    if height_offset and display_rows + 5 >= max_h:
        del v_items[-(cols * height_offset):]
    if self.docstring is None:
        self.list_win.resize(rows + 2, w)
    else:
        docstring = self.format_docstring(self.docstring, max_w - 2, max_h - height_offset)
        docstring_string = ''.join(docstring)
        rows += len(docstring)
        self.list_win.resize(rows, max_w)
    if down:
        self.list_win.mvwin(y + 1, 0)
    else:
        self.list_win.mvwin(y - rows - 2, 0)
    if v_items:
        self.list_win.addstr('\n ')
    for ix, i in enumerate(v_items):
        padding = (wl - len(i)) * ' '
        if i == current_item:
            color = get_colpair(self.config, 'operator')
        else:
            color = get_colpair(self.config, 'main')
        self.list_win.addstr(i + padding, color)
        if (cols == 1 or (ix and (not (ix + 1) % cols))) and ix + 1 < len(v_items):
            self.list_win.addstr('\n ')
    if self.docstring is not None:
        self.list_win.addstr('\n' + docstring_string, get_colpair(self.config, 'comment'))
    y = self.list_win.getyx()[0]
    self.list_win.resize(y + 2, w)
    self.statusbar.win.touchwin()
    self.statusbar.win.noutrefresh()
    self.list_win.attron(get_colpair(self.config, 'main'))
    self.list_win.border()
    self.scr.touchwin()
    self.scr.cursyncup()
    self.scr.noutrefresh()
    self.scr.move(*self.scr.getyx())
    self.list_win.refresh()