import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def prde_no_cancel_b_small(b, Q, n, DE):
    """
    Parametric Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], n in ZZ, and b, q1, ..., qm in k[t] with
    deg(b) < deg(D) - 1 and either D == d/dt or deg(D) >= 2, returns
    h1, ..., hr in k[t] and a matrix A with coefficients in Const(k) such that
    if c1, ..., cm in Const(k) and q in k[t] satisfy deg(q) <= n and
    Dq + b*q == Sum(ci*qi, (i, 1, m)) then q = Sum(dj*hj, (j, 1, r)) where
    d1, ..., dr in Const(k) and A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0.
    """
    m = len(Q)
    H = [Poly(0, DE.t)] * m
    for N, i in itertools.product(range(n, 0, -1), range(m)):
        si = Q[i].nth(N + DE.d.degree(DE.t) - 1) / (N * DE.d.LC())
        sitn = Poly(si * DE.t ** N, DE.t)
        H[i] = H[i] + sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b * sitn
    if b.degree(DE.t) > 0:
        for i in range(m):
            si = Poly(Q[i].nth(b.degree(DE.t)) / b.LC(), DE.t)
            H[i] = H[i] + si
            Q[i] = Q[i] - derivation(si, DE) - b * si
        if all((qi.is_zero for qi in Q)):
            dc = -1
        else:
            dc = max([qi.degree(DE.t) for qi in Q])
        M = Matrix(dc + 1, m, lambda i, j: Q[j].nth(i), DE.t)
        A, u = constant_system(M, zeros(dc + 1, 1, DE.t), DE)
        c = eye(m, DE.t)
        A = A.row_join(zeros(A.rows, m, DE.t)).col_join(c.row_join(-c))
        return (H, A)
    t = DE.t
    if DE.case != 'base':
        with DecrementLevel(DE):
            t0 = DE.t
            ba, bd = frac_in(b, t0, field=True)
            Q0 = [frac_in(qi.TC(), t0, field=True) for qi in Q]
            f, B = param_rischDE(ba, bd, Q0, DE)
        f = [Poly(fa.as_expr() / fd.as_expr(), t, field=True) for fa, fd in f]
        B = Matrix.from_Matrix(B.to_Matrix(), t)
    else:
        f = [Poly(1, t, field=True)]
        B = Matrix([[qi.TC() for qi in Q] + [S.Zero]], DE.t)
    d = max([qi.degree(DE.t) for qi in Q])
    if d > 0:
        M = Matrix(d, m, lambda i, j: Q[j].nth(i + 1), DE.t)
        A, _ = constant_system(M, zeros(d, 1, DE.t), DE)
    else:
        A = Matrix(0, m, [], DE.t)
    r = len(f)
    I = eye(m, DE.t)
    A = A.row_join(zeros(A.rows, r + m, DE.t))
    B = B.row_join(zeros(B.rows, m, DE.t))
    C = I.row_join(zeros(m, r, DE.t)).row_join(-I)
    return (f + H, A.col_join(B).col_join(C))