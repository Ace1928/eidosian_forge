from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan
from sympy.polys.polyroots import roots
from sympy.polys.polytools import cancel
from sympy.polys.rootoftools import RootSum
from sympy.polys import Poly, resultant, ZZ
def log_to_real(h, q, x, t):
    """
    Convert complex logarithms to real functions.

    Explanation
    ===========

    Given real field K and polynomials h in K[t,x] and q in K[t],
    returns real function f such that:
                          ___
                  df   d  \\  `
                  -- = --  )  a log(h(a, x))
                  dx   dx /__,
                         a | q(a) = 0

    Examples
    ========

        >>> from sympy.integrals.rationaltools import log_to_real
        >>> from sympy.abc import x, y
        >>> from sympy import Poly, S
        >>> log_to_real(Poly(x + 3*y/2 + S(1)/2, x, domain='QQ[y]'),
        ... Poly(3*y**2 + 1, y, domain='ZZ'), x, y)
        2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3
        >>> log_to_real(Poly(x**2 - 1, x, domain='ZZ'),
        ... Poly(-2*y + 1, y, domain='ZZ'), x, y)
        log(x**2 - 1)/2

    See Also
    ========

    log_to_atan
    """
    from sympy.simplify.radsimp import collect
    u, v = symbols('u,v', cls=Dummy)
    H = h.as_expr().subs({t: u + I * v}).expand()
    Q = q.as_expr().subs({t: u + I * v}).expand()
    H_map = collect(H, I, evaluate=False)
    Q_map = collect(Q, I, evaluate=False)
    a, b = (H_map.get(S.One, S.Zero), H_map.get(I, S.Zero))
    c, d = (Q_map.get(S.One, S.Zero), Q_map.get(I, S.Zero))
    R = Poly(resultant(c, d, v), u)
    R_u = roots(R, filter='R')
    if len(R_u) != R.count_roots():
        return None
    result = S.Zero
    for r_u in R_u.keys():
        C = Poly(c.subs({u: r_u}), v)
        R_v = roots(C, filter='R')
        if len(R_v) != C.count_roots():
            return None
        R_v_paired = []
        for r_v in R_v:
            if r_v not in R_v_paired and -r_v not in R_v_paired:
                if r_v.is_negative or r_v.could_extract_minus_sign():
                    R_v_paired.append(-r_v)
                elif not r_v.is_zero:
                    R_v_paired.append(r_v)
        for r_v in R_v_paired:
            D = d.subs({u: r_u, v: r_v})
            if D.evalf(chop=True) != 0:
                continue
            A = Poly(a.subs({u: r_u, v: r_v}), x)
            B = Poly(b.subs({u: r_u, v: r_v}), x)
            AB = (A ** 2 + B ** 2).as_expr()
            result += r_u * log(AB) + r_v * log_to_atan(A, B)
    R_q = roots(q, filter='R')
    if len(R_q) != q.count_roots():
        return None
    for r in R_q.keys():
        result += r * log(h.as_expr().subs(t, r))
    return result