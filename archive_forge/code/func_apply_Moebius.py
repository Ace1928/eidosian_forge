from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def apply_Moebius(m, z):
    """
    Applies the matrix m to the ideal point z::

        sage: from sage.all import matrix, CIF, RIF
        sage: m = matrix([[CIF(2,1), CIF(4,2)], [CIF(2,3), CIF(4,2)]])
        sage: apply_Moebius(m, CIF(3,4)) # doctest: +NUMERIC12
        0.643835616438356? - 0.383561643835617?*I
        sage: apply_Moebius(m, Infinity) # doctest: +NUMERIC12
        0.5384615384615385? - 0.3076923076923078?*I

    """
    if isinstance(m, ExtendedMatrix):
        if m.isOrientationReversing and z != Infinity:
            z = z.conjugate()
        m = m.matrix
    if m[0, 0] == 1 and m[1, 1] == 1 and (m[0, 1] == 0) and (m[1, 0] == 0):
        return z
    if z == Infinity:
        return m[0, 0] / m[1, 0]
    return (m[0, 0] * z + m[0, 1]) / (m[1, 0] * z + m[1, 1])