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
def scroll_buffer(self, up: bool=True, reset: bool=False, lines: int | None=None) -> None:
    """
        Scroll the scrolling buffer up (up=True) or down (up=False) the given
        amount of lines or half the screen height.

        If just 'reset' is True, set the scrollbuffer view to the current
        terminal content.
        """
    if reset:
        self.scrolling_up = 0
        self.set_term_cursor()
        return
    if lines is None:
        lines = self.height // 2
    if not up:
        lines = -lines
    maxscroll = len(self.scrollback_buffer)
    self.scrolling_up += lines
    if self.scrolling_up > maxscroll:
        self.scrolling_up = maxscroll
    elif self.scrolling_up < 0:
        self.scrolling_up = 0
    self.set_term_cursor()