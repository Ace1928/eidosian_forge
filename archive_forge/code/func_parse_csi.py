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
def parse_csi(self, char: bytes) -> None:
    """
        Parse ECMA-48 CSI (Control Sequence Introducer) sequences.
        """
    qmark = self.escbuf.startswith(b'?')
    escbuf = []
    for arg in self.escbuf[1 if qmark else 0:].split(b';'):
        try:
            num = int(arg)
        except ValueError:
            num = None
        escbuf.append(num)
    cmd_ = CSI_COMMANDS[char]
    if cmd_ is not None:
        if isinstance(cmd_, CSIAlias):
            csi_cmd: CSICommand = CSI_COMMANDS[cmd_.alias]
        elif isinstance(cmd_, CSICommand):
            csi_cmd = cmd_
        elif cmd_[0] == 'alias':
            csi_cmd = CSI_COMMANDS[CSIAlias(*cmd_).alias]
        else:
            csi_cmd = CSICommand(*cmd_)
        number_of_args, default_value, cmd = csi_cmd
        while len(escbuf) < number_of_args:
            escbuf.append(default_value)
        for i in range(len(escbuf)):
            if escbuf[i] is None or escbuf[i] == 0:
                escbuf[i] = default_value
        with suppress(ValueError):
            cmd(self, escbuf, qmark)