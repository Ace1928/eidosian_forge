from operator import add, neg, pos, sub, mul
from collections import defaultdict
from sympy.utilities.iterables import _strongly_connected_components
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from .ddm import DDM
from .lll import ddm_lll, ddm_lll_transform
from sympy.polys.domains import QQ
def sdm_nullspace_from_rref(A, one, ncols, pivots, nonzero_cols):
    """Get nullspace from A which is in RREF"""
    nonpivots = sorted(set(range(ncols)) - set(pivots))
    K = []
    for j in nonpivots:
        Kj = {j: one}
        for i in nonzero_cols.get(j, ()):
            Kj[pivots[i]] = -A[i][j]
        K.append(Kj)
    return (K, nonpivots)