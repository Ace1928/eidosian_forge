from ..matrix import vector, matrix
from ..math_basics import is_RealIntervalFieldElement
from ..sage_helper import _within_sage
from a real type (either a SnapPy.Number or one
def space_r13_normalise(u):
    """
    Given a space-like vector in Minkowski space, returns the normalised
    vector (with norm 1).
    """
    return u / r13_dot(u, u).sqrt()