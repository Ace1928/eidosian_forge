import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def hermediv(c1, c2):
    """
    Divide one Hermite series by another.

    Returns the quotient-with-remainder of two Hermite series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Hermite series coefficients representing the quotient and
        remainder.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermemul, hermepow

    Notes
    -----
    In general, the (polynomial) division of one Hermite series by another
    results in quotient and remainder terms that are not in the Hermite
    polynomial basis set.  Thus, to express these results as a Hermite
    series, it is necessary to "reproject" the results onto the Hermite
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermediv
    >>> hermediv([ 14.,  15.,  28.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))
    >>> hermediv([ 15.,  17.,  28.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([1., 2.]))

    """
    return pu._div(hermemul, c1, c2)