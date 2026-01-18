from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def linear_palette(palette: Palette, n: int) -> Palette:
    """ Generate a new palette as a subset of a given palette.

    Given an input ``palette``, take ``n`` colors from it by dividing its
    length into ``n`` (approximately) evenly spaced indices.

    Args:

        palette (seq[str]) : a sequence of hex RGB color strings
        n (int) : the size of the output palette to generate

    Returns:
        seq[str] : a sequence of hex RGB color strings

    Raises:
        ValueError if n > len(palette)

    """
    if n > len(palette):
        raise ValueError(f"Requested {n} colors, function can only return colors up to the base palette's length ({len(palette)})")
    return tuple((palette[int(math.floor(i))] for i in np.linspace(0, len(palette) - 1, num=n)))