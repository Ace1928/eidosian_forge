from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit
def interp2d_kernel(x, xp, fp, left, right, output_y):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        xval = x[i, j]
        if isnan(xval):
            output_y[i, j] = nan
        elif xval < xp[0]:
            output_y[i, j] = left
        elif xval >= xp[-1]:
            output_y[i, j] = right
        else:
            upper_i = len(xp) - 1
            lower_i = 0
            while True:
                stop_i = 1 + (lower_i + upper_i) // 2
                if xp[stop_i] < xval:
                    lower_i = stop_i
                elif xp[stop_i - 1] > xval:
                    upper_i = stop_i - 1
                else:
                    break
            x0 = xp[stop_i - 1]
            x1 = xp[stop_i]
            y0 = fp[stop_i - 1]
            y1 = fp[stop_i]
            slope = (y1 - y0) / (x1 - x0)
            y_interp = y0 + slope * (xval - x0)
            output_y[i, j] = y_interp