from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def cosh_dist(self, other):
    """
        Returns cosh of the distance of this finite point to another
        finite point::

            sage: from sage.all import *
            sage: a = FinitePoint(CIF(1,2),RIF(3))
            sage: b = FinitePoint(CIF(4,5),RIF(6))
            sage: a.cosh_dist(b) # doctest: +NUMERIC12
            1.7500000000000000?

        """
    r = 1 + ((self.t - other.t) ** 2 + _abs_sqr(self.z - other.z)) / (2 * self.t * other.t)
    RIF = r.parent()
    if _within_sage:
        if is_RealIntervalFieldElement(r):
            return r.intersection(RIF(1, sage.all.Infinity))
    if r < 1.0:
        return RIF(1.0)
    return r