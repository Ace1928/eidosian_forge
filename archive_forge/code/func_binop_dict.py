from operator import add, neg, pos, sub, mul
from collections import defaultdict
from sympy.utilities.iterables import _strongly_connected_components
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from .ddm import DDM
from .lll import ddm_lll, ddm_lll_transform
from sympy.polys.domains import QQ
def binop_dict(A, B, fab, fa, fb):
    Anz, Bnz = (set(A), set(B))
    C = {}
    for i in Anz & Bnz:
        Ai, Bi = (A[i], B[i])
        Ci = {}
        Anzi, Bnzi = (set(Ai), set(Bi))
        for j in Anzi & Bnzi:
            Cij = fab(Ai[j], Bi[j])
            if Cij:
                Ci[j] = Cij
        for j in Anzi - Bnzi:
            Cij = fa(Ai[j])
            if Cij:
                Ci[j] = Cij
        for j in Bnzi - Anzi:
            Cij = fb(Bi[j])
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci
    for i in Anz - Bnz:
        Ai = A[i]
        Ci = {}
        for j, Aij in Ai.items():
            Cij = fa(Aij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci
    for i in Bnz - Anz:
        Bi = B[i]
        Ci = {}
        for j, Bij in Bi.items():
            Cij = fb(Bij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci
    return C