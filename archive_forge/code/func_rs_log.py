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
def rs_log(p, x, prec):
    """
    The Logarithm of ``p`` modulo ``O(x**prec)``.

    Notes
    =====

    Truncation of ``integral dx p**-1*d p/dx`` is used.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_log
    >>> R, x = ring('x', QQ)
    >>> rs_log(1 + x, x, 8)
    1/7*x**7 - 1/6*x**6 + 1/5*x**5 - 1/4*x**4 + 1/3*x**3 - 1/2*x**2 + x
    >>> rs_log(x**QQ(3, 2) + 1, x, 5)
    1/3*x**(9/2) - 1/2*x**3 + x**(3/2)
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_log, p, x, prec)
    R = p.ring
    if p == 1:
        return R.zero
    c = _get_constant_term(p, x)
    if c:
        const = 0
        if c == 1:
            pass
        else:
            c_expr = c.as_expr()
            if R.domain is EX:
                const = log(c_expr)
            elif isinstance(c, PolyElement):
                try:
                    const = R(log(c_expr))
                except ValueError:
                    R = R.add_gens([log(c_expr)])
                    p = p.set_ring(R)
                    x = x.set_ring(R)
                    c = c.set_ring(R)
                    const = R(log(c_expr))
            else:
                try:
                    const = R(log(c))
                except ValueError:
                    raise DomainError('The given series cannot be expanded in this domain.')
        dlog = p.diff(x)
        dlog = rs_mul(dlog, _series_inversion1(p, x, prec), x, prec - 1)
        return rs_integrate(dlog, x) + const
    else:
        raise NotImplementedError