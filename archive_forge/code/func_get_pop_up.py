from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
def get_pop_up(self) -> tuple[int, int, tuple[Widget, int, int]] | None:
    c = self.coords.get('pop up', None)
    if not c:
        return None
    return c