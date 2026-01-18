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
@property
def base_widget(self) -> Widget:
    """Read-only property that steps through decoration widgets and returns the one at the base.

        This default implementation returns self.
        """
    return self