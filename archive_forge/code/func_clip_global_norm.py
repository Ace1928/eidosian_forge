import os
import sys
import hashlib
import uuid
import warnings
import collections
import weakref
import requests
import numpy as np
from .. import ndarray
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def clip_global_norm(arrays, max_norm, check_isfinite=True):
    """Rescales NDArrays so that the sum of their 2-norm is smaller than `max_norm`.

    Parameters
    ----------
    arrays : list of NDArray
    max_norm : float
    check_isfinite : bool, default True
         If True, check that the total_norm is finite (not nan or inf). This
         requires a blocking .asscalar() call.

    Returns
    -------
    NDArray or float
      Total norm. Return type is NDArray of shape (1,) if check_isfinite is
      False. Otherwise a float is returned.

    """

    def _norm(array):
        if array.stype == 'default':
            x = array.reshape((-1,))
            return ndarray.dot(x, x)
        return array.norm().square()
    assert len(arrays) > 0
    ctx = arrays[0].context
    total_norm = ndarray.add_n(*[_norm(arr).as_in_context(ctx) for arr in arrays])
    total_norm = ndarray.sqrt(total_norm)
    if check_isfinite:
        if not np.isfinite(total_norm.asscalar()):
            warnings.warn(UserWarning('nan or inf is detected. Clipping results will be undefined.'), stacklevel=2)
    scale = max_norm / (total_norm + 1e-08)
    scale = ndarray.min(ndarray.concat(scale, ndarray.ones(1, ctx=ctx), dim=0))
    for arr in arrays:
        arr *= scale.as_in_context(arr.context)
    if check_isfinite:
        return total_norm.asscalar()
    else:
        return total_norm