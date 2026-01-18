from __future__ import annotations
from math import floor as mfloor
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError
def mu_small(k: int, j: int) -> bool:
    return abs(mu[k][j]) <= half