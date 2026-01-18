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
def csi_set_scroll(self, top: int=0, bottom: int=0) -> None:
    """
        Set scrolling region, 'top' is the line number of first line in the
        scrolling region. 'bottom' is the line number of bottom line. If both
        are set to 0, the whole screen will be used (default).
        """
    if not top:
        top = 1
    if not bottom:
        bottom = self.height
    if top < bottom <= self.height:
        self.scrollregion_start = self.constrain_coords(0, top - 1, ignore_scrolling=True)[1]
        self.scrollregion_end = self.constrain_coords(0, bottom - 1, ignore_scrolling=True)[1]
        self.set_term_cursor(0, 0)