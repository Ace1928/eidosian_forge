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
def set_tabstop(self, x: int | None=None, remove: bool=False, clear: bool=False) -> None:
    if clear:
        for tab in range(len(self.tabstops)):
            self.tabstops[tab] = 0
        return
    if x is None:
        x = self.term_cursor[0]
    div, mod = divmod(x, 8)
    if remove:
        self.tabstops[div] &= ~(1 << mod)
    else:
        self.tabstops[div] |= 1 << mod