from __future__ import annotations
from math import floor as mfloor
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError
def closest_integer(x):
    return ZZ(mfloor(x + half))