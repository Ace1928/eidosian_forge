import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """Crop src at fixed location, and (optionally) resize it to size.

    Parameters
    ----------
    src : NDArray
        Input image
    x0 : int
        Left boundary of the cropping area
    y0 : int
        Top boundary of the cropping area
    w : int
        Width of the cropping area
    h : int
        Height of the cropping area
    size : tuple of (w, h)
        Optional, resize to new size after cropping
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.

    Returns
    -------
    NDArray
        An `NDArray` containing the cropped image.
    """
    out = src[y0:y0 + h, x0:x0 + w]
    if size is not None and (w, h) != size:
        sizes = (h, w, size[1], size[0])
        out = imresize(out, *size, interp=_get_interp_method(interp, sizes))
    return out