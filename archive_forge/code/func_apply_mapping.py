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
def apply_mapping(self, char: bytes) -> bytes:
    if self._sgr_mapping or self._g[self.active] == 'ibmpc':
        dec_pos = DEC_SPECIAL_CHARS.find(char.decode('cp437'))
        if dec_pos >= 0:
            self.current = '0'
            return ALT_DEC_SPECIAL_CHARS[dec_pos].encode('cp437')
        self.current = 'U'
        return char
    return char