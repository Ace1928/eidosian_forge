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
def moments_weighted_normalized(self):
    mu = self.moments_weighted_central
    if self._multichannel:
        nchannels = self._intensity_image.shape[-1]
        return np.stack([_moments.moments_normalized(mu[..., i], order=3, spacing=self._spacing) for i in range(nchannels)], axis=-1)
    else:
        return _moments.moments_normalized(mu, order=3, spacing=self._spacing)