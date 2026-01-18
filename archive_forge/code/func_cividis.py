from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def cividis(n: int) -> Palette:
    """ Generate a palette of colors from the Cividis palette.

    The full Cividis palette that serves as input for deriving new palettes
    has 256 colors, and looks like:

    :bokeh-palette:`cividis(256)`

    Args:
        n (int) : size of the palette to generate

    Returns:
        seq[str] : a sequence of hex RGB color strings

    Raises:
        ValueError if n is greater than the base palette length of 256

    Examples:

    .. code-block:: python

        >>> cividis(6)
        ('#00204C', '#31446B', '#666870', '#958F78', '#CAB969', '#FFE945')

    The resulting palette looks like: :bokeh-palette:`cividis(6)`

    """
    return linear_palette(Cividis256, n)