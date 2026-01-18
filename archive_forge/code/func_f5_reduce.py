from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
def f5_reduce(f, B):
    """
    F5-reduce a labeled polynomial f by B.

    Continuously searches for non-zero labeled polynomial h in B, such
    that the leading term lt_h of h divides the leading term lt_f of
    f and Sign(lt_h * h) < Sign(f). If such a labeled polynomial h is
    found, f gets replaced by f - lt_f / lt_h * h. If no such h can be
    found or f is 0, f is no further F5-reducible and f gets returned.

    A polynomial that is reducible in the usual sense need not be
    F5-reducible, e.g.:

    >>> from sympy.polys.groebnertools import lbp, sig, f5_reduce, Polyn
    >>> from sympy.polys import ring, QQ, lex

    >>> R, x,y,z = ring("x,y,z", QQ, lex)

    >>> f = lbp(sig((1, 1, 1), 4), x, 3)
    >>> g = lbp(sig((0, 0, 0), 2), x, 2)

    >>> Polyn(f).rem([Polyn(g)])
    0
    >>> f5_reduce(f, [g])
    (((1, 1, 1), 4), x, 3)

    """
    order = Polyn(f).ring.order
    domain = Polyn(f).ring.domain
    if not Polyn(f):
        return f
    while True:
        g = f
        for h in B:
            if Polyn(h):
                if monomial_divides(Polyn(h).LM, Polyn(f).LM):
                    t = term_div(Polyn(f).LT, Polyn(h).LT, domain)
                    if sig_cmp(sig_mult(Sign(h), t[0]), Sign(f), order) < 0:
                        hp = lbp_mul_term(h, t)
                        f = lbp_sub(f, hp)
                        break
        if g == f or not Polyn(f):
            return f