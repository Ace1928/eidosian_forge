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
def csi_set_attr(self, attrs: Sequence[int]) -> None:
    """
        Set graphics rendition.
        """
    if attrs[-1] == 0:
        self.attrspec = None
    attributes = set()
    if self.attrspec is None:
        fg = bg = None
    else:
        if 'default' in self.attrspec.foreground:
            fg = None
        else:
            fg = self.attrspec.foreground_number
            if fg >= 8 and self.attrspec.colors == 16:
                fg -= 8
        if 'default' in self.attrspec.background:
            bg = None
        else:
            bg = self.attrspec.background_number
            if bg >= 8 and self.attrspec.colors == 16:
                bg -= 8
        for attr in ('bold', 'underline', 'blink', 'standout'):
            if not getattr(self.attrspec, attr):
                continue
            attributes.add(attr)
    attrspec = self.sgi_to_attrspec(attrs, fg, bg, attributes, self.attrspec.colors if self.attrspec else 1)
    if self.modes.reverse_video:
        self.attrspec = self.reverse_attrspec(attrspec)
    else:
        self.attrspec = attrspec