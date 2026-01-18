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
def csi_status_report(self, mode: Literal[5, 6]) -> None:
    """
        Report various information about the terminal status.
        Information is queried by 'mode', where possible values are:
            5 -> device status report
            6 -> cursor position report
        """
    if mode == 5:
        self.widget.respond(f'{ESC}[0n')
    elif mode == 6:
        x, y = self.term_cursor
        self.widget.respond(ESC + f'[{y + 1:d};{x + 1:d}R')