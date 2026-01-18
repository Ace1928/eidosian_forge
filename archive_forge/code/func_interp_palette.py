from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def interp_palette(palette: Palette, n: int) -> Palette:
    """ Generate a new palette by interpolating a given palette.

    Linear interpolation is performed separately on each of the RGBA
    components.

    Args:
        palette (seq[str]) :
            A sequence of hex RGB(A) color strings to create new palette from

        n (int) :
            The size of the palette to generate

    Returns:
        tuple[str] : a sequence of hex RGB(A) color strings

    Raises:
        ValueError if ``n`` is negative or the supplied ``palette`` is empty.

    """
    npalette = len(palette)
    if npalette < 1:
        raise ValueError('palette must contain at least one color')
    if n < 0:
        raise ValueError('requested palette length cannot be negative')
    rgba_array = to_rgba_array(palette)
    integers = np.arange(npalette)
    fractions = np.linspace(0, npalette - 1, n)
    r = np.interp(fractions, integers, rgba_array[:, 0]).astype(np.uint8)
    g = np.interp(fractions, integers, rgba_array[:, 1]).astype(np.uint8)
    b = np.interp(fractions, integers, rgba_array[:, 2]).astype(np.uint8)
    a = np.interp(fractions, integers, rgba_array[:, 3]) / 255.0
    return tuple((RGB(*args).to_hex() for args in zip(r, g, b, a)))