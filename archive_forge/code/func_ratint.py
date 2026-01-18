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
def ratint(f, x, **flags):
    """
    Performs indefinite integration of rational functions.

    Explanation
    ===========

    Given a field :math:`K` and a rational function :math:`f = p/q`,
    where :math:`p` and :math:`q` are polynomials in :math:`K[x]`,
    returns a function :math:`g` such that :math:`f = g'`.

    Examples
    ========

    >>> from sympy.integrals.rationaltools import ratint
    >>> from sympy.abc import x

    >>> ratint(36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), x)
    (12*x + 6)/(x**2 - 1) + 4*log(x - 2) - 4*log(x + 1)

    References
    ==========

    .. [1] M. Bronstein, Symbolic Integration I: Transcendental
       Functions, Second Edition, Springer-Verlag, 2005, pp. 35-70

    See Also
    ========

    sympy.integrals.integrals.Integral.doit
    sympy.integrals.rationaltools.ratint_logpart
    sympy.integrals.rationaltools.ratint_ratpart

    """
    if isinstance(f, tuple):
        p, q = f
    else:
        p, q = f.as_numer_denom()
    p, q = (Poly(p, x, composite=False, field=True), Poly(q, x, composite=False, field=True))
    coeff, p, q = p.cancel(q)
    poly, p = p.div(q)
    result = poly.integrate(x).as_expr()
    if p.is_zero:
        return coeff * result
    g, h = ratint_ratpart(p, q, x)
    P, Q = h.as_numer_denom()
    P = Poly(P, x)
    Q = Poly(Q, x)
    q, r = P.div(Q)
    result += g + q.integrate(x).as_expr()
    if not r.is_zero:
        symbol = flags.get('symbol', 't')
        if not isinstance(symbol, Symbol):
            t = Dummy(symbol)
        else:
            t = symbol.as_dummy()
        L = ratint_logpart(r, Q, x, t)
        real = flags.get('real')
        if real is None:
            if isinstance(f, tuple):
                p, q = f
                atoms = p.atoms() | q.atoms()
            else:
                atoms = f.atoms()
            for elt in atoms - {x}:
                if not elt.is_extended_real:
                    real = False
                    break
            else:
                real = True
        eps = S.Zero
        if not real:
            for h, q in L:
                _, h = h.primitive()
                eps += RootSum(q, Lambda(t, t * log(h.as_expr())), quadratic=True)
        else:
            for h, q in L:
                _, h = h.primitive()
                R = log_to_real(h, q, x, t)
                if R is not None:
                    eps += R
                else:
                    eps += RootSum(q, Lambda(t, t * log(h.as_expr())), quadratic=True)
        result += eps
    return coeff * result