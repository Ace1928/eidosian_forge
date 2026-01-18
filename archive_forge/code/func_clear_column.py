from collections import defaultdict
from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ
def clear_column(m):
    if m[0][0] == 0:
        return m
    pivot = m[0][0]
    for j in range(1, rows):
        if m[j][0] == 0:
            continue
        d, r = domain.div(m[j][0], pivot)
        if r == 0:
            add_rows(m, 0, j, 1, 0, -d, 1)
        else:
            a, b, g = domain.gcdex(pivot, m[j][0])
            d_0 = domain.div(m[j][0], g)[0]
            d_j = domain.div(pivot, g)[0]
            add_rows(m, 0, j, a, b, d_0, -d_j)
            pivot = g
    return m