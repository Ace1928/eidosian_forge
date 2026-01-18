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
def set_term_cursor(self, x: int | None=None, y: int | None=None) -> None:
    """
        Set terminal cursor to x/y and update canvas cursor. If one or both axes
        are omitted, use the values of the current position.
        """
    if x is None:
        x = self.term_cursor[0]
    if y is None:
        y = self.term_cursor[1]
    self.term_cursor = self.constrain_coords(x, y)
    if self.has_focus and self.modes.visible_cursor and (self.scrolling_up < self.height - y):
        self.cursor = (x, y + self.scrolling_up)
    else:
        self.cursor = None