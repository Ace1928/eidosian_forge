from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def copy_modified(self, fg: str | None=None, bg: str | None=None, colors: Literal[1, 16, 88, 256, 16777216] | None=None) -> Self:
    if fg is None:
        foreground = self.foreground
    else:
        foreground = fg
    if bg is None:
        background = self.background
    else:
        background = bg
    if colors is None:
        new_colors = self.colors
    else:
        new_colors = colors
    return self.__class__(foreground, background, new_colors)