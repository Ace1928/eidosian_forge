from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def setMappingMode(self, mapping):
    """
        Sets the way that values outside of the range 0 to 1 are mapped to colors.

        Parameters
        ----------
        mapping: int or str
            Sets mapping mode to

            - `ColorMap.CLIP` or 'clip': Values are clipped to the range 0 to 1. ColorMap defaults to this.
            - `ColorMap.REPEAT` or 'repeat': Colors repeat cyclically, i.e. range 1 to 2 repeats the colors for 0 to 1.
            - `ColorMap.MIRROR` or 'mirror': The range 0 to -1 uses same colors (in reverse order) as 0 to 1.
            - `ColorMap.DIVERGING` or 'diverging': Colors are mapped to -1 to 1 such that the central value appears at 0.
        """
    if isinstance(mapping, str):
        mapping = self.enumMap[mapping.lower()]
    if mapping in [self.CLIP, self.REPEAT, self.DIVERGING, self.MIRROR]:
        self.mapping_mode = mapping
    else:
        raise ValueError(f"Undefined mapping type '{mapping}'")
    self.stopsCache = {}