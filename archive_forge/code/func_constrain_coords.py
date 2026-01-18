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
def constrain_coords(self, x: int, y: int, ignore_scrolling: bool=False) -> tuple[int, int]:
    """
        Checks if x/y are within the terminal and returns the corrected version.
        If 'ignore_scrolling' is set, constrain within the full size of the
        screen and not within scrolling region.
        """
    if x >= self.width:
        x = self.width - 1
    elif x < 0:
        x = 0
    if self.modes.constrain_scrolling and (not ignore_scrolling):
        if y > self.scrollregion_end:
            y = self.scrollregion_end
        elif y < self.scrollregion_start:
            y = self.scrollregion_start
    elif y >= self.height:
        y = self.height - 1
    elif y < 0:
        y = 0
    return (x, y)