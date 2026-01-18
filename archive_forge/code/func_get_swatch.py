from __future__ import annotations
import typing
from dataclasses import dataclass
from functools import cached_property
from .._colormaps import PaletteInterpolatedMap
from .._colormaps._colormap import ColorMapKind
def get_swatch(self, num_colors: int) -> RGB256Swatch:
    """
        Get a swatch with given number of colors
        """
    index = num_colors - self.min_colors
    return self.swatches[index]