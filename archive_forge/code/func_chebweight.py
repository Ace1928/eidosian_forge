import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def chebweight(x):
    """
    The weight function of the Chebyshev polynomials.

    The weight function is :math:`1/\\sqrt{1 - x^2}` and the interval of
    integration is :math:`[-1, 1]`. The Chebyshev polynomials are
    orthogonal, but not normalized, with respect to this weight function.

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
    w = 1.0 / (np.sqrt(1.0 + x) * np.sqrt(1.0 - x))
    return w