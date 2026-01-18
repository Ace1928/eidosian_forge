from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def point_at_t(self, t):
    """
        Evaluate the curve at a single point, returning a tuple of *d* floats.
        """
    return tuple(self(t))