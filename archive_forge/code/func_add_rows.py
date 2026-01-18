from collections import defaultdict
from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ
def add_rows(m, i, j, a, b, c, d):
    for k in range(cols):
        e = m[i][k]
        m[i][k] = a * e + b * m[j][k]
        m[j][k] = c * e + d * m[j][k]