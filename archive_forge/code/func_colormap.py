from __future__ import annotations
import typing
from dataclasses import dataclass
from functools import cached_property
from .._colormaps import PaletteInterpolatedMap
from .._colormaps._colormap import ColorMapKind
@cached_property
def colormap(self) -> PaletteInterpolatedMap:
    """
        Return a colormap representation of the palette
        """
    return PaletteInterpolatedMap(self)