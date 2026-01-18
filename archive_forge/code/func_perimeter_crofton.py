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
@only2d
def perimeter_crofton(self):
    if len(np.unique(self._spacing)) != 1:
        raise NotImplementedError('`perimeter` supports isotropic spacings only')
    return perimeter_crofton(self.image, 4) * self._spacing[0]