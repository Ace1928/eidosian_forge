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
def rs_sinh(p, x, prec):
    """
    Hyperbolic sine of a series

    Return the series expansion of the sinh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_sinh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_sinh(x + x*y, x, 4)
    1/6*x**3*y**3 + 1/2*x**3*y**2 + 1/2*x**3*y + 1/6*x**3 + x*y + x

    See Also
    ========

    sinh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sinh, p, x, prec)
    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t - t1) / 2