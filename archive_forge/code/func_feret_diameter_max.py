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
def feret_diameter_max(self):
    identity_convex_hull = np.pad(self.image_convex, 2, mode='constant', constant_values=0)
    if self._ndim == 2:
        coordinates = np.vstack(find_contours(identity_convex_hull, 0.5, fully_connected='high'))
    elif self._ndim == 3:
        coordinates, _, _, _ = marching_cubes(identity_convex_hull, level=0.5)
    distances = pdist(coordinates * self._spacing, 'sqeuclidean')
    return sqrt(np.max(distances))