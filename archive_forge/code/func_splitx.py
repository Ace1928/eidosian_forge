import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def splitx(self, *args):
    """
        Return a list of new `Bbox` objects formed by splitting the original
        one with vertical lines at fractional positions given by *args*.
        """
    xf = [0, *args, 1]
    x0, y0, x1, y1 = self.extents
    w = x1 - x0
    return [Bbox([[x0 + xf0 * w, y0], [x0 + xf1 * w, y1]]) for xf0, xf1 in zip(xf[:-1], xf[1:])]