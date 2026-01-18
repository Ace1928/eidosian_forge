import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def num_obs_y(Y):
    """
    Return the number of original observations that correspond to a
    condensed distance matrix.

    Parameters
    ----------
    Y : array_like
        Condensed distance matrix.

    Returns
    -------
    n : int
        The number of observations in the condensed distance matrix `Y`.

    Examples
    --------
    Find the number of original observations corresponding to a
    condensed distance matrix Y.
    
    >>> from scipy.spatial.distance import num_obs_y
    >>> Y = [1, 2, 3.5, 7, 10, 4]
    >>> num_obs_y(Y)
    4
    """
    Y = np.asarray(Y, order='c')
    is_valid_y(Y, throw=True, name='Y')
    k = Y.shape[0]
    if k == 0:
        raise ValueError('The number of observations cannot be determined on an empty distance matrix.')
    d = int(np.ceil(np.sqrt(k * 2)))
    if d * (d - 1) / 2 != k:
        raise ValueError('Invalid condensed distance matrix passed. Must be some k where k=(n choose 2) for some n >= 2.')
    return d