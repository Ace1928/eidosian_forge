from __future__ import annotations
import atexit
import copy
import errno
import fcntl
import os
import pty
import selectors
import signal
import struct
import sys
import termios
import time
import traceback
import typing
import warnings
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from urwid import event_loop, util
from urwid.canvas import Canvas
from urwid.display import AttrSpec, RealTerminal
from urwid.display.escape import ALT_DEC_SPECIAL_CHARS, DEC_SPECIAL_CHARS
from urwid.widget import Sizing, Widget
from .display.common import _BASIC_COLORS, _color_desc_256, _color_desc_true
def insert_chars(self, position: tuple[int, int] | None=None, chars: int=1, char: bytes | None=None) -> None:
    """
        Insert 'chars' number of either empty characters - or those specified by
        'char' - before 'position' (or the current position if not specified)
        pushing subsequent characters of the line to the right without wrapping.
        """
    if position is None:
        position = self.term_cursor
    if chars == 0:
        chars = 1
    if char is None:
        char_spec = self.empty_char()
    else:
        char_spec = (self.attrspec, self.charset.current, char)
    x, y = position
    while chars > 0:
        self.term[y].insert(x, char_spec)
        self.term[y].pop()
        chars -= 1