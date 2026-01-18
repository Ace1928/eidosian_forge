import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def legweight(x):
    """
    Weight function of the Legendre polynomials.

    The weight function is :math:`1` and the interval of integration is
    :math:`[-1, 1]`. The Legendre polynomials are orthogonal, but not
    normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    w = x * 0.0 + 1.0
    return w