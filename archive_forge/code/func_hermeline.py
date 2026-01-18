import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def hermeline(off, scl):
    """
    Hermite series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Hermite series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeline
    >>> from numpy.polynomial.hermite_e import hermeline, hermeval
    >>> hermeval(0,hermeline(3, 2))
    3.0
    >>> hermeval(1,hermeline(3, 2))
    5.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])