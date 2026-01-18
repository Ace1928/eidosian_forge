from __future__ import annotations
from math import floor as mfloor
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError
def lovasz_condition(k: int) -> bool:
    return g_star[k] >= (delta - mu[k][k - 1] ** 2) * g_star[k - 1]