import snappy
from sage.all import ZZ, PolynomialRing
def make_rect_eqn(R, A, B, c, vars_per_tet=1):
    n = len(A)
    R_vars = R.gens()
    if vars_per_tet == 2:
        Z, W = (R_vars[:n], R_vars[n:])
    else:
        Z = R_vars
    left, right = (R(1), R(1))
    for i, a in enumerate(A):
        term = Z[i] ** abs(a)
        if a > 0:
            left *= term
        elif a < 0:
            right *= term
    for i, b in enumerate(B):
        if vars_per_tet == 1:
            term = (1 - Z[i]) ** abs(b)
        else:
            term = W[i] ** abs(b)
        if b > 0:
            left *= term
        elif b < 0:
            right *= term
    return left - c * right