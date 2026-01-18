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
def set_g01(self, char: bytes, mod: bytes) -> None:
    """
        Set G0 or G1 according to 'char' and modifier 'mod'.
        """
    if self.modes.main_charset != CHARSET_DEFAULT:
        return
    if mod == b'(':
        g = 0
    else:
        g = 1
    if char == b'0':
        cset = 'vt100'
    elif char == b'U':
        cset = 'ibmpc'
    elif char == b'K':
        cset = 'user'
    else:
        cset = 'default'
    self.charset.define(g, cset)