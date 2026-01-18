from __future__ import annotations
import numba as nb
import numpy as np
import os
@nb.jit('(uint32,)', nopython=True, nogil=True, cache=True)
def extract_scaled(x):
    """Extract components as float64 values in [0.0, 1.0]"""
    r = np.float64((x & 255) / 255)
    g = np.float64((x >> 8 & 255) / 255)
    b = np.float64((x >> 16 & 255) / 255)
    a = np.float64((x >> 24 & 255) / 255)
    return (r, g, b, a)