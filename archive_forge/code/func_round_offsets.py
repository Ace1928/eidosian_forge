import collections
from collections.abc import Iterable
import functools
import math
import warnings
from affine import Affine
import attr
import numpy as np
from rasterio.errors import WindowError, RasterioDeprecationWarning
from rasterio.transform import rowcol, guard_transform
def round_offsets(self, **kwds):
    """Return a copy with column and row offsets rounded.

        Offsets are rounded to the preceding whole number. The lengths
        are not changed.

        Parameters
        ----------
        kwds : dict
            Collects keyword arguments that are no longer used.

        Returns
        -------
        Window

        """
    operator = lambda x: int(math.floor(x + 0.001))
    row_off = operator(self.row_off)
    col_off = operator(self.col_off)
    return Window(col_off, row_off, self.width, self.height)