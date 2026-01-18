from operator import add, neg, pos, sub, mul
from collections import defaultdict
from sympy.utilities.iterables import _strongly_connected_components
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from .ddm import DDM
from .lll import ddm_lll, ddm_lll_transform
from sympy.polys.domains import QQ
def sdm_particular_from_rref(A, ncols, pivots):
    """Get a particular solution from A which is in RREF"""
    P = {}
    for i, j in enumerate(pivots):
        Ain = A[i].get(ncols - 1, None)
        if Ain is not None:
            P[j] = Ain / A[i][j]
    return P