import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
def infer_interval_breaks(self, centers):
    breaks = infer_interval_breaks(centers, axis=1)
    breaks = infer_interval_breaks(breaks, axis=0)
    return breaks