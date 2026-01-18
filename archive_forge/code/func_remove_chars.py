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
def remove_chars(self, position: tuple[int, int] | None=None, chars: int=1) -> None:
    """
        Remove 'chars' number of empty characters from 'position' (or the current
        position if not specified) pulling subsequent characters of the line to
        the left without joining any subsequent lines.
        """
    if position is None:
        position = self.term_cursor
    if chars == 0:
        chars = 1
    x, y = position
    while chars > 0:
        self.term[y].pop(x)
        self.term[y].append(self.empty_char())
        chars -= 1