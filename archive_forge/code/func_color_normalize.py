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
def color_normalize(src, mean, std=None):
    """Normalize src with mean and std.

    Parameters
    ----------
    src : NDArray
        Input image
    mean : NDArray
        RGB mean to be subtracted
    std : NDArray
        RGB standard deviation to be divided

    Returns
    -------
    NDArray
        An `NDArray` containing the normalized image.
    """
    if mean is not None:
        src -= mean
    if std is not None:
        src /= std
    return src