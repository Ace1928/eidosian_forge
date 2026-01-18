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
def image_intensity(self):
    if self._intensity_image is None:
        raise AttributeError('No intensity image specified.')
    image = self.image if not self._multichannel else np.expand_dims(self.image, self._ndim)
    return self._intensity_image[self.slice] * image