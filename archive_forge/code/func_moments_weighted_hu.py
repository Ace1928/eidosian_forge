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
def moments_weighted_hu(self):
    if not (np.array(self._spacing) == np.array([1, 1])).all():
        raise NotImplementedError('`moments_hu` supports spacing = (1, 1) only')
    nu = self.moments_weighted_normalized
    if self._multichannel:
        nchannels = self._intensity_image.shape[-1]
        return np.stack([_moments.moments_hu(nu[..., i]) for i in range(nchannels)], axis=-1)
    else:
        return _moments.moments_hu(nu)