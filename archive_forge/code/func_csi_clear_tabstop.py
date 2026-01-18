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
def csi_clear_tabstop(self, mode: Literal[0, 3]=0):
    """
        Clear tabstop at current position or if 'mode' is 3, delete all
        tabstops.
        """
    if mode == 0:
        self.set_tabstop(remove=True)
    elif mode == 3:
        self.set_tabstop(clear=True)