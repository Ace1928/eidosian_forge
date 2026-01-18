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
@_cached
def image_filled(self):
    structure = np.ones((3,) * self._ndim)
    return ndi.binary_fill_holes(self.image, structure)