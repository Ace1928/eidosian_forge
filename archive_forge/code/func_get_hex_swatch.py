from __future__ import annotations
import typing
from dataclasses import dataclass
from functools import cached_property
from .._colormaps import PaletteInterpolatedMap
from .._colormaps._colormap import ColorMapKind
def get_hex_swatch(self, num_colors: int) -> RGBHexSwatch:
    """
        Get a swatch with given number of colors in hex
        """
    swatch = self.get_swatch(num_colors)
    return RGB256Swatch_to_RGBHexSwatch(swatch)