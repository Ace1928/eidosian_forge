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
def is_blank_row(row: list[tuple[object, Literal['0', 'U'] | None], bytes]) -> bool:
    if len(row) > 1:
        return False
    if row[0][2].strip():
        return False
    return True