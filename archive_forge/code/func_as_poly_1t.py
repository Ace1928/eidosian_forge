from the names used in Bronstein's book.
from types import GeneratorType
from functools import reduce
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, cos,
from .integrals import integrate, Integral
from .heurisch import _symbols
from sympy.polys.polyerrors import DomainError, PolynomialError
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,
from sympy.polys.rootoftools import RootSum
from sympy.utilities.iterables import numbered_symbols
def as_poly_1t(p, t, z):
    """
    (Hackish) way to convert an element ``p`` of K[t, 1/t] to K[t, z].

    In other words, ``z == 1/t`` will be a dummy variable that Poly can handle
    better.

    See issue 5131.

    Examples
    ========

    >>> from sympy import random_poly
    >>> from sympy.integrals.risch import as_poly_1t
    >>> from sympy.abc import x, z

    >>> p1 = random_poly(x, 10, -10, 10)
    >>> p2 = random_poly(x, 10, -10, 10)
    >>> p = p1 + p2.subs(x, 1/x)
    >>> as_poly_1t(p, x, z).as_expr().subs(z, 1/x) == p
    True
    """
    pa, pd = frac_in(p, t, cancel=True)
    if not pd.is_monomial:
        raise PolynomialError('%s is not an element of K[%s, 1/%s].' % (p, t, t))
    d = pd.degree(t)
    one_t_part = pa.slice(0, d + 1)
    r = pd.degree() - pa.degree()
    t_part = pa - one_t_part
    try:
        t_part = t_part.to_field().exquo(pd)
    except DomainError as e:
        raise NotImplementedError(e)
    one_t_part = Poly.from_list(reversed(one_t_part.rep.rep), *one_t_part.gens, domain=one_t_part.domain)
    if 0 < r < oo:
        one_t_part *= Poly(t ** r, t)
    one_t_part = one_t_part.replace(t, z)
    if pd.nth(d):
        one_t_part *= Poly(1 / pd.nth(d), z, expand=False)
    ans = t_part.as_poly(t, z, expand=False) + one_t_part.as_poly(t, z, expand=False)
    return ans