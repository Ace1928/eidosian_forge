from __future__ import annotations
import functools
import logging
import typing
import warnings
from operator import attrgetter
from urwid import signals
from urwid.canvas import Canvas, CanvasCache, CompositeCanvas
from urwid.command_map import command_map
from urwid.split_repr import split_repr
from urwid.util import MetaSuper
from .constants import Sizing
def fixed_size(size: tuple[()]) -> None:
    """
    raise ValueError if size != ().

    Used by FixedWidgets to test size parameter.
    """
    if size != ():
        raise ValueError(f'FixedWidget takes only () for size.passed: {size!r}')