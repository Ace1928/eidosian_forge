from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def plasma(n: int) -> Palette:
    """ Generate a palette of colors from the Plasma palette.

    The full Plasma palette that serves as input for deriving new palettes
    has 256 colors, and looks like:

    :bokeh-palette:`plasma(256)`

    Args:
        n (int) : size of the palette to generate

    Returns:
        seq[str] : a sequence of hex RGB color strings

    Raises:
        ValueError if n is greater than the base palette length of 256

    Examples:

    .. code-block:: python

        >>> plasma(6)
        ('#0C0786', '#6A00A7', '#B02A8F', '#E06461', '#FCA635', '#EFF821')

    The resulting palette looks like: :bokeh-palette:`plasma(6)`

    """
    return linear_palette(Plasma256, n)