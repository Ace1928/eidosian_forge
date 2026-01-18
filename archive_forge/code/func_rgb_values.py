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
def rgb_values(n) -> tuple[int | None, int | None, int | None]:
    if colors == 16:
        aspec = AttrSpec(f'h{n:d}', '', 256)
    else:
        aspec = AttrSpec(f'h{n:d}', '', colors)
    return aspec.get_rgb_values()[:3]