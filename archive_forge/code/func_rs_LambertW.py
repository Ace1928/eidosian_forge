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
def rs_LambertW(p, x, prec):
    """
    Calculate the series expansion of the principal branch of the Lambert W
    function.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_LambertW
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_LambertW(x + x*y, x, 3)
    -x**2*y**2 - 2*x**2*y - x**2 + x*y + x

    See Also
    ========

    LambertW
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_LambertW, p, x, prec)
    R = p.ring
    p1 = R(0)
    if _has_constant_term(p, x):
        raise NotImplementedError('Polynomial must not have constant term in the series variables')
    if x in R.gens:
        for precx in _giant_steps(prec):
            e = rs_exp(p1, x, precx)
            p2 = rs_mul(e, p1, x, precx) - p
            p3 = rs_mul(e, p1 + 1, x, precx)
            p3 = rs_series_inversion(p3, x, precx)
            tmp = rs_mul(p2, p3, x, precx)
            p1 -= tmp
        return p1
    else:
        raise NotImplementedError