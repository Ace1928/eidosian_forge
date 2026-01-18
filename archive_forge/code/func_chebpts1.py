import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def chebpts1(npts):
    """
    Chebyshev points of the first kind.

    The Chebyshev points of the first kind are the points ``cos(x)``,
    where ``x = [pi*(k + .5)/npts for k in range(npts)]``.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the first kind.

    See Also
    --------
    chebpts2

    Notes
    -----

    .. versionadded:: 1.5.0

    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError('npts must be integer')
    if _npts < 1:
        raise ValueError('npts must be >= 1')
    x = 0.5 * np.pi / _npts * np.arange(-_npts + 1, _npts + 1, 2)
    return np.sin(x)