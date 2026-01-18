from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational, igcd
from sympy.functions import sin, cos, tan, atan, exp, atanh, tanh, log, ceiling
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math
def rs_tan(p, x, prec):
    """
    Tangent of a series.

    Return the series expansion of the tan of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_tan
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_tan(x + x*y, x, 4)
    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x

   See Also
   ========

   _tan1, tan
   """
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_tan, p, x, prec)
        return r
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            const = tan(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(tan(c_expr))
            except ValueError:
                R = R.add_gens([tan(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                const = R(tan(c_expr))
        else:
            try:
                const = R(tan(c))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        p1 = p - c
        t2 = rs_tan(p1, x, prec)
        t = rs_series_inversion(1 - const * t2, x, prec)
        return rs_mul(const + t2, t, x, prec)
    if R.ngens == 1:
        return _tan1(p, x, prec)
    else:
        return rs_fun(p, rs_tan, x, prec)