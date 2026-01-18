import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def legline(off, scl):
    """
    Legendre series whose graph is a straight line.



    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Legendre series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> import numpy.polynomial.legendre as L
    >>> L.legline(3,2)
    array([3, 2])
    >>> L.legval(-3, L.legline(3,2)) # should be -3
    -3.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])