import platform
import re
from colorsys import rgb_to_hls
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple
from ._palettes import EIGHT_BIT_PALETTE, STANDARD_PALETTE, WINDOWS_PALETTE
from .color_triplet import ColorTriplet
from .repr import Result, rich_repr
from .terminal_theme import DEFAULT_TERMINAL_THEME
def blend_rgb(color1: ColorTriplet, color2: ColorTriplet, cross_fade: float=0.5) -> ColorTriplet:
    """Blend one RGB color in to another."""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    new_color = ColorTriplet(int(r1 + (r2 - r1) * cross_fade), int(g1 + (g2 - g1) * cross_fade), int(b1 + (b2 - b1) * cross_fade))
    return new_color