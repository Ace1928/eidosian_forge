from operator import add, neg, pos, sub, mul
from collections import defaultdict
from sympy.utilities.iterables import _strongly_connected_components
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from .ddm import DDM
from .lll import ddm_lll, ddm_lll_transform
from sympy.polys.domains import QQ
def sdm_transpose(M):
    MT = {}
    for i, Mi in M.items():
        for j, Mij in Mi.items():
            try:
                MT[j][i] = Mij
            except KeyError:
                MT[j] = {i: Mij}
    return MT