from functools import wraps
import numpy as np
from patsy.util import (atleast_2d_column_default,
def memorize_chunk(self, x, center=True, rescale=True, ddof=0):
    x = atleast_2d_column_default(x)
    if self.current_mean is None:
        self.current_mean = np.zeros(x.shape[1], dtype=wide_dtype_for(x))
        self.current_M2 = np.zeros(x.shape[1], dtype=wide_dtype_for(x))
    for i in range(x.shape[0]):
        self.current_n += 1
        delta = x[i, :] - self.current_mean
        self.current_mean += delta / self.current_n
        self.current_M2 += delta * (x[i, :] - self.current_mean)