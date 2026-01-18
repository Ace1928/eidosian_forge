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
def nocache_widget_render_instance(self):
    """
    Return a function that wraps the cls.render() method
    and finalizes the canvas that it returns, but does not
    cache the canvas.
    """
    fn = self.render.original_fn

    @functools.wraps(fn)
    def finalize_render(size, focus=False):
        canv = fn(self, size, focus=focus)
        if canv.widget_info:
            canv = CompositeCanvas(canv)
        canv.finalize(self, size, focus)
        return canv
    finalize_render.original_fn = fn
    return finalize_render