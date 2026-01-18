from __future__ import annotations
import typing
from copy import copy
import numpy as np
from ..iapi import panel_ranges
def dist_euclidean(x: FloatArrayLike, y: FloatArrayLike) -> FloatArray:
    """
    Calculate euclidean distance
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2, dtype=np.float64)