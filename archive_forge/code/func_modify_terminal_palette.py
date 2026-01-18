from __future__ import annotations
import abc
import contextlib
import functools
import os
import platform
import selectors
import signal
import socket
import sys
import typing
from urwid import signals, str_util, util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, UPDATE_PALETTE_ENTRY, AttrSpec, BaseScreen, RealTerminal
def modify_terminal_palette(self, entries: list[tuple[int, int | None, int | None, int | None]]):
    """
        entries - list of (index, red, green, blue) tuples.

        Attempt to set part of the terminal palette (this does not work
        on all terminals.)  The changes are sent as a single escape
        sequence so they should all take effect at the same time.

        0 <= index < 256 (some terminals will only have 16 or 88 colors)
        0 <= red, green, blue < 256
        """
    if self.term == 'fbterm':
        modify = [f'{index:d};{red:d};{green:d};{blue:d}' for index, red, green, blue in entries]
        self.write(f'\x1b[3;{';'.join(modify)}}}')
    else:
        modify = [f'{index:d};rgb:{red:02x}/{green:02x}/{blue:02x}' for index, red, green, blue in entries]
        self.write(f'\x1b]4;{';'.join(modify)}\x1b\\')
    self.flush()