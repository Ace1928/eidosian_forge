from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from ._interpolated import InterpolatedMap
def discrete_palette(self, n):
    """
        Pick exact colors from the swatch if possible
        """
    if n <= self.palette.max_colors:
        return self.palette.get_hex_swatch(n)
    return super().discrete_palette(n)