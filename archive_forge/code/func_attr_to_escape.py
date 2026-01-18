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
def attr_to_escape(a: AttrSpec | str | None) -> str:
    if a in self._pal_escape:
        return self._pal_escape[a]
    if isinstance(a, AttrSpec):
        return self._attrspec_to_escape(a)
    if a is None:
        return self._attrspec_to_escape(AttrSpec('default', 'default'))
    self.logger.debug(f'Undefined attribute: {a!r}')
    return self._attrspec_to_escape(AttrSpec('default', 'default'))