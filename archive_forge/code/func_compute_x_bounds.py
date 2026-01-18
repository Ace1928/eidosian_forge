import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
def compute_x_bounds(self, xr_ds):
    x_breaks = self.infer_interval_breaks(xr_ds[self.x].values)
    bounds = Glyph._compute_bounds_2d(x_breaks)
    return self.maybe_expand_bounds(bounds)