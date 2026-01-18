import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import (
@property
def centroid_local(self):
    M = self.moments
    M0 = M[(0,) * self._ndim]

    def _get_element(axis):
        return (0,) * axis + (1,) + (0,) * (self._ndim - 1 - axis)
    return np.asarray(tuple((M[_get_element(axis)] / M0 for axis in range(self._ndim))))