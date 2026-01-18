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
def set_cursor_home() -> str:
    if not partial_display():
        return escape.set_cursor_position(0, 0)
    return escape.CURSOR_HOME_COL + escape.move_cursor_up(cy)