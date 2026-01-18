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
def moments_weighted_central(self):
    ctr = self.centroid_weighted_local
    image = self._image_intensity_double()
    if self._multichannel:
        moments_list = [_moments.moments_central(image[..., i], center=ctr[..., i], order=3, spacing=self._spacing) for i in range(image.shape[-1])]
        moments = np.stack(moments_list, axis=-1)
    else:
        moments = _moments.moments_central(image, ctr, order=3, spacing=self._spacing)
    return moments