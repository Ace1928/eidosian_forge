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
def rs_atan(p, x, prec):
    """
    The arctangent of a series

    Return the series expansion of the atan of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_atan
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_atan(x + x*y, x, 4)
    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x

    See Also
    ========

    atan
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atan, p, x, prec)
    R = p.ring
    const = 0
    if _has_constant_term(p, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = atan(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(atan(c_expr))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        else:
            try:
                const = R(atan(c))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
    dp = p.diff(x)
    p1 = rs_square(p, x, prec) + R(1)
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const