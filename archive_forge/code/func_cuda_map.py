import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
def cuda_map(in_array):
    out_array = cupy.zeros(in_array.shape, dtype='float64')
    in_array = cuda.to_device(in_array)
    kernel[cuda_args(in_array.shape)](in_array, out_array)
    return out_array