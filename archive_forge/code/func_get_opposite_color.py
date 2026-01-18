from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
@memoized()
def get_opposite_color(colorname: str | None) -> str | None:
    """
    Take a color name in either 'ansi...' format or 6 digit RGB, return the
    color of opposite luminosity (same hue/saturation).

    This is used for turning color schemes that work on a light background
    usable on a dark background.
    """
    if colorname is None:
        return None
    if colorname in ('', 'default'):
        return colorname
    try:
        return OPPOSITE_ANSI_COLOR_NAMES[colorname]
    except KeyError:
        r = int(colorname[:2], 16) / 255.0
        g = int(colorname[2:4], 16) / 255.0
        b = int(colorname[4:6], 16) / 255.0
        h, l, s = rgb_to_hls(r, g, b)
        l = 1 - l
        r, g, b = hls_to_rgb(h, l, s)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        return f'{r:02x}{g:02x}{b:02x}'