import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def parametric_log_deriv_heu(fa, fd, wa, wd, DE, c1=None):
    """
    Parametric logarithmic derivative heuristic.

    Explanation
    ===========

    Given a derivation D on k[t], f in k(t), and a hyperexponential monomial
    theta over k(t), raises either NotImplementedError, in which case the
    heuristic failed, or returns None, in which case it has proven that no
    solution exists, or returns a solution (n, m, v) of the equation
    n*f == Dv/v + m*Dtheta/theta, with v in k(t)* and n, m in ZZ with n != 0.

    If this heuristic fails, the structure theorem approach will need to be
    used.

    The argument w == Dtheta/theta
    """
    c1 = c1 or Dummy('c1')
    p, a = fa.div(fd)
    q, b = wa.div(wd)
    B = max(0, derivation(DE.t, DE).degree(DE.t) - 1)
    C = max(p.degree(DE.t), q.degree(DE.t))
    if q.degree(DE.t) > B:
        eqs = [p.nth(i) - c1 * q.nth(i) for i in range(B + 1, C + 1)]
        s = solve(eqs, c1)
        if not s or not s[c1].is_Rational:
            return None
        M, N = s[c1].as_numer_denom()
        M_poly = M.as_poly(q.gens)
        N_poly = N.as_poly(q.gens)
        nfmwa = N_poly * fa * wd - M_poly * wa * fd
        nfmwd = fd * wd
        Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE, 'auto')
        if Qv is None:
            return None
        Q, v = Qv
        if Q.is_zero or v.is_zero:
            return None
        return (Q * N, Q * M, v)
    if p.degree(DE.t) > B:
        return None
    c = lcm(fd.as_poly(DE.t).LC(), wd.as_poly(DE.t).LC())
    l = fd.monic().lcm(wd.monic()) * Poly(c, DE.t)
    ln, ls = splitfactor(l, DE)
    z = ls * ln.gcd(ln.diff(DE.t))
    if not z.has(DE.t):
        return None
    u1, r1 = (fa * l.quo(fd)).div(z)
    u2, r2 = (wa * l.quo(wd)).div(z)
    eqs = [r1.nth(i) - c1 * r2.nth(i) for i in range(z.degree(DE.t))]
    s = solve(eqs, c1)
    if not s or not s[c1].is_Rational:
        return None
    M, N = s[c1].as_numer_denom()
    nfmwa = N.as_poly(DE.t) * fa * wd - M.as_poly(DE.t) * wa * fd
    nfmwd = fd * wd
    Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE)
    if Qv is None:
        return None
    Q, v = Qv
    if Q.is_zero or v.is_zero:
        return None
    return (Q * N, Q * M, v)