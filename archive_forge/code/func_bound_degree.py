from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
def bound_degree(a, b, cQ, DE, case='auto', parametric=False):
    """
    Bound on polynomial solutions.

    Explanation
    ===========

    Given a derivation D on k[t] and ``a``, ``b``, ``c`` in k[t] with ``a != 0``, return
    n in ZZ such that deg(q) <= n for any solution q in k[t] of
    a*Dq + b*q == c, when parametric=False, or deg(q) <= n for any solution
    c1, ..., cm in Const(k) and q in k[t] of a*Dq + b*q == Sum(ci*gi, (i, 1, m))
    when parametric=True.

    For ``parametric=False``, ``cQ`` is ``c``, a ``Poly``; for ``parametric=True``, ``cQ`` is Q ==
    [q1, ..., qm], a list of Polys.

    This constitutes step 3 of the outline given in the rde.py docstring.
    """
    if case == 'auto':
        case = DE.case
    da = a.degree(DE.t)
    db = b.degree(DE.t)
    if parametric:
        dc = max([i.degree(DE.t) for i in cQ])
    else:
        dc = cQ.degree(DE.t)
    alpha = cancel(-b.as_poly(DE.t).LC().as_expr() / a.as_poly(DE.t).LC().as_expr())
    if case == 'base':
        n = max(0, dc - max(db, da - 1))
        if db == da - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)
    elif case == 'primitive':
        if db > da:
            n = max(0, dc - db)
        else:
            n = max(0, dc - da + 1)
        etaa, etad = frac_in(DE.d, DE.T[DE.level - 1])
        t1 = DE.t
        with DecrementLevel(DE):
            alphaa, alphad = frac_in(alpha, DE.t)
            if db == da - 1:
                from .prde import limited_integrate
                try:
                    (za, zd), m = limited_integrate(alphaa, alphad, [(etaa, etad)], DE)
                except NonElementaryIntegralException:
                    pass
                else:
                    if len(m) != 1:
                        raise ValueError('Length of m should be 1')
                    n = max(n, m[0])
            elif db == da:
                from .prde import is_log_deriv_k_t_radical_in_field
                A = is_log_deriv_k_t_radical_in_field(alphaa, alphad, DE)
                if A is not None:
                    aa, z = A
                    if aa == 1:
                        beta = -(a * derivation(z, DE).as_poly(t1) + b * z.as_poly(t1)).LC() / (z.as_expr() * a.LC())
                        betaa, betad = frac_in(beta, DE.t)
                        from .prde import limited_integrate
                        try:
                            (za, zd), m = limited_integrate(betaa, betad, [(etaa, etad)], DE)
                        except NonElementaryIntegralException:
                            pass
                        else:
                            if len(m) != 1:
                                raise ValueError('Length of m should be 1')
                            n = max(n, m[0].as_expr())
    elif case == 'exp':
        from .prde import parametric_log_deriv
        n = max(0, dc - max(db, da))
        if da == db:
            etaa, etad = frac_in(DE.d.quo(Poly(DE.t, DE.t)), DE.T[DE.level - 1])
            with DecrementLevel(DE):
                alphaa, alphad = frac_in(alpha, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    a, m, z = A
                    if a == 1:
                        n = max(n, m)
    elif case in ('tan', 'other_nonlinear'):
        delta = DE.d.degree(DE.t)
        lam = DE.d.LC()
        alpha = cancel(alpha / lam)
        n = max(0, dc - max(da + delta - 1, db))
        if db == da + delta - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', 'other_nonlinear', 'base'}, not %s." % case)
    return n