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
def axis_minor_length(self):
    if self._ndim == 2:
        l2 = self.inertia_tensor_eigvals[-1]
        return 4 * sqrt(l2)
    elif self._ndim == 3:
        ev = self.inertia_tensor_eigvals
        return sqrt(10 * (-ev[0] + ev[1] + ev[2]))
    else:
        raise ValueError('axis_minor_length only available in 2D and 3D')