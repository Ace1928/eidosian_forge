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
def csi_set_keyboard_leds(self, mode: Literal[0, 1, 2, 3]=0) -> None:
    """
        Set keyboard LEDs, modes are:
            0 -> clear all LEDs
            1 -> set scroll lock LED
            2 -> set num lock LED
            3 -> set caps lock LED

        This currently just emits a signal, so it can be processed by another
        widget or the main application.
        """
    states = {0: 'clear', 1: 'scroll_lock', 2: 'num_lock', 3: 'caps_lock'}
    if mode in states:
        self.widget.leds(states[mode])