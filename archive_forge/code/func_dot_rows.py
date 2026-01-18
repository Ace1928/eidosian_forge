from __future__ import annotations
from math import floor as mfloor
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError
def dot_rows(x, y, rows: tuple[int, int]):
    return sum([x[rows[0]][z] * y[rows[1]][z] for z in range(x.shape[1])])