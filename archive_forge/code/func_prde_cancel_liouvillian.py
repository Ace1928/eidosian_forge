import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def prde_cancel_liouvillian(b, Q, n, DE):
    """
    Pg, 237.
    """
    H = []
    if DE.case == 'primitive':
        with DecrementLevel(DE):
            ba, bd = frac_in(b, DE.t, field=True)
    for i in range(n, -1, -1):
        if DE.case == 'exp':
            with DecrementLevel(DE):
                ba, bd = frac_in(b + (i * (derivation(DE.t, DE) / DE.t)).as_poly(b.gens), DE.t, field=True)
        with DecrementLevel(DE):
            Qy = [frac_in(q.nth(i), DE.t, field=True) for q in Q]
            fi, Ai = param_rischDE(ba, bd, Qy, DE)
        fi = [Poly(fa.as_expr() / fd.as_expr(), DE.t, field=True) for fa, fd in fi]
        Ai = Ai.set_gens(DE.t)
        ri = len(fi)
        if i == n:
            M = Ai
        else:
            M = Ai.col_join(M.row_join(zeros(M.rows, ri, DE.t)))
        Fi, hi = ([None] * ri, [None] * ri)
        for j in range(ri):
            hji = fi[j] * (DE.t ** i).as_poly(fi[j].gens)
            hi[j] = hji
            Fi[j] = -(derivation(hji, DE) - b * hji)
        H += hi
        Q = Q + Fi
    return (H, M)