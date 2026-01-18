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
def is_valid_dm(D, tol=0.0, throw=False, name='D', warning=False):
    """
    Return True if input array is a valid distance matrix.

    Distance matrices must be 2-dimensional numpy arrays.
    They must have a zero-diagonal, and they must be symmetric.

    Parameters
    ----------
    D : array_like
        The candidate object to test for validity.
    tol : float, optional
        The distance matrix should be symmetric. `tol` is the maximum
        difference between entries ``ij`` and ``ji`` for the distance
        metric to be considered symmetric.
    throw : bool, optional
        An exception is thrown if the distance matrix passed is not valid.
    name : str, optional
        The name of the variable to checked. This is useful if
        throw is set to True so the offending variable can be identified
        in the exception message when an exception is thrown.
    warning : bool, optional
        Instead of throwing an exception, a warning message is
        raised.

    Returns
    -------
    valid : bool
        True if the variable `D` passed is a valid distance matrix.

    Notes
    -----
    Small numerical differences in `D` and `D.T` and non-zeroness of
    the diagonal are ignored if they are within the tolerance specified
    by `tol`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import is_valid_dm

    This matrix is a valid distance matrix.

    >>> d = np.array([[0.0, 1.1, 1.2, 1.3],
    ...               [1.1, 0.0, 1.0, 1.4],
    ...               [1.2, 1.0, 0.0, 1.5],
    ...               [1.3, 1.4, 1.5, 0.0]])
    >>> is_valid_dm(d)
    True

    In the following examples, the input is not a valid distance matrix.

    Not square:

    >>> is_valid_dm([[0, 2, 2], [2, 0, 2]])
    False

    Nonzero diagonal element:

    >>> is_valid_dm([[0, 1, 1], [1, 2, 3], [1, 3, 0]])
    False

    Not symmetric:

    >>> is_valid_dm([[0, 1, 3], [2, 0, 1], [3, 1, 0]])
    False

    """
    D = np.asarray(D, order='c')
    valid = True
    try:
        s = D.shape
        if len(D.shape) != 2:
            if name:
                raise ValueError("Distance matrix '%s' must have shape=2 (i.e. be two-dimensional)." % name)
            else:
                raise ValueError('Distance matrix must have shape=2 (i.e. be two-dimensional).')
        if tol == 0.0:
            if not (D == D.T).all():
                if name:
                    raise ValueError("Distance matrix '%s' must be symmetric." % name)
                else:
                    raise ValueError('Distance matrix must be symmetric.')
            if not (D[range(0, s[0]), range(0, s[0])] == 0).all():
                if name:
                    raise ValueError("Distance matrix '%s' diagonal must be zero." % name)
                else:
                    raise ValueError('Distance matrix diagonal must be zero.')
        else:
            if not (D - D.T <= tol).all():
                if name:
                    raise ValueError(f"Distance matrix '{name}' must be symmetric within tolerance {tol:5.5f}.")
                else:
                    raise ValueError('Distance matrix must be symmetric within tolerance %5.5f.' % tol)
            if not (D[range(0, s[0]), range(0, s[0])] <= tol).all():
                if name:
                    raise ValueError(f"Distance matrix '{name}' diagonal must be close to zero within tolerance {tol:5.5f}.")
                else:
                    raise ValueError("Distance matrix '{}' diagonal must be close to zero within tolerance {:5.5f}.".format(*tol))
    except Exception as e:
        if throw:
            raise
        if warning:
            warnings.warn(str(e), stacklevel=2)
        valid = False
    return valid