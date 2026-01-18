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
def get_utf8_len(self, bytenum: int) -> int:
    """
        Process startbyte and return the number of bytes following it to get a
        valid UTF-8 multibyte sequence.

        bytenum -- an integer ordinal
        """
    length = 0
    while bytenum & 64:
        bytenum <<= 1
        length += 1
    return length