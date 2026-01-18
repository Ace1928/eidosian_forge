from __future__ import annotations
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.utilities.misc import as_int
def continued_fraction(a) -> list:
    """Return the continued fraction representation of a Rational or
    quadratic irrational.

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction
    >>> from sympy import sqrt
    >>> continued_fraction((1 + 2*sqrt(3))/5)
    [0, 1, [8, 3, 34, 3]]

    See Also
    ========
    continued_fraction_periodic, continued_fraction_reduce, continued_fraction_convergents
    """
    e = _sympify(a)
    if all((i.is_Rational for i in e.atoms())):
        if e.is_Integer:
            return continued_fraction_periodic(e, 1, 0)
        elif e.is_Rational:
            return continued_fraction_periodic(e.p, e.q, 0)
        elif e.is_Pow and e.exp is S.Half and e.base.is_Integer:
            return continued_fraction_periodic(0, 1, e.base)
        elif e.is_Mul and len(e.args) == 2 and (e.args[0].is_Rational and e.args[1].is_Pow and e.args[1].base.is_Integer and (e.args[1].exp is S.Half)):
            a, b = e.args
            return continued_fraction_periodic(0, a.q, b.base, a.p)
        else:
            p, d = e.expand().as_numer_denom()
            if d.is_Integer:
                if p.is_Rational:
                    return continued_fraction_periodic(p, d)
                if p.is_Add and len(p.args) == 2:
                    a, bc = p.args
                else:
                    a = S.Zero
                    bc = p
                if a.is_Integer:
                    b = S.NaN
                    if bc.is_Mul and len(bc.args) == 2:
                        b, c = bc.args
                    elif bc.is_Pow:
                        b = Integer(1)
                        c = bc
                    if b.is_Integer and (c.is_Pow and c.exp is S.Half and c.base.is_Integer):
                        c = c.base
                        return continued_fraction_periodic(a, d, c, b)
    raise ValueError('expecting a rational or quadratic irrational, not %s' % e)