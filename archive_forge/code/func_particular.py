from operator import add, neg, pos, sub, mul
from collections import defaultdict
from sympy.utilities.iterables import _strongly_connected_components
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from .ddm import DDM
from .lll import ddm_lll, ddm_lll_transform
from sympy.polys.domains import QQ
def particular(A):
    ncols = A.shape[1]
    B, pivots, nzcols = sdm_irref(A)
    P = sdm_particular_from_rref(B, ncols, pivots)
    rep = {0: P} if P else {}
    return A.new(rep, (1, ncols - 1), A.domain)