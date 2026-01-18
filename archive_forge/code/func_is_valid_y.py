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
def is_valid_y(y, warning=False, throw=False, name=None):
    """
    Return True if the input array is a valid condensed distance matrix.

    Condensed distance matrices must be 1-dimensional numpy arrays.
    Their length must be a binomial coefficient :math:`{n \\choose 2}`
    for some positive integer n.

    Parameters
    ----------
    y : array_like
        The condensed distance matrix.
    warning : bool, optional
        Invokes a warning if the variable passed is not a valid
        condensed distance matrix. The warning message explains why
        the distance matrix is not valid.  `name` is used when
        referencing the offending variable.
    throw : bool, optional
        Throws an exception if the variable passed is not a valid
        condensed distance matrix.
    name : bool, optional
        Used when referencing the offending variable in the
        warning or exception message.

    Returns
    -------
    bool
        True if the input array is a valid condensed distance matrix,
        False otherwise.

    Examples
    --------
    >>> from scipy.spatial.distance import is_valid_y

    This vector is a valid condensed distance matrix.  The length is 6,
    which corresponds to ``n = 4``, since ``4*(4 - 1)/2`` is 6.

    >>> v = [1.0, 1.2, 1.0, 0.5, 1.3, 0.9]
    >>> is_valid_y(v)
    True

    An input vector with length, say, 7, is not a valid condensed distance
    matrix.

    >>> is_valid_y([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    False

    """
    y = np.asarray(y, order='c')
    valid = True
    try:
        if len(y.shape) != 1:
            if name:
                raise ValueError("Condensed distance matrix '%s' must have shape=1 (i.e. be one-dimensional)." % name)
            else:
                raise ValueError('Condensed distance matrix must have shape=1 (i.e. be one-dimensional).')
        n = y.shape[0]
        d = int(np.ceil(np.sqrt(n * 2)))
        if d * (d - 1) / 2 != n:
            if name:
                raise ValueError("Length n of condensed distance matrix '%s' must be a binomial coefficient, i.e.there must be a k such that (k \\choose 2)=n)!" % name)
            else:
                raise ValueError('Length n of condensed distance matrix must be a binomial coefficient, i.e. there must be a k such that (k \\choose 2)=n)!')
    except Exception as e:
        if throw:
            raise
        if warning:
            warnings.warn(str(e), stacklevel=2)
        valid = False
    return valid