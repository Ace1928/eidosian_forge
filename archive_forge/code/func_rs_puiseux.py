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
def rs_puiseux(f, p, x, prec):
    """
    Return the puiseux series for `f(p, x, prec)`.

    To be used when function ``f`` is implemented only for regular series.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_puiseux, rs_exp
    >>> R, x = ring('x', QQ)
    >>> p = x**QQ(2,5) + x**QQ(2,3) + x
    >>> rs_puiseux(rs_exp,p, x, 1)
    1/2*x**(4/5) + x**(2/3) + x**(2/5) + 1
    """
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = int(n * den // igcd(n, den))
        elif power != int(power):
            den = power.denominator
            n = int(n * den // igcd(n, den))
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, x, prec * n)
        n1 = QQ(1, n)
        if isinstance(r, tuple):
            r = tuple([pow_xin(rx, index, n1) for rx in r])
        else:
            r = pow_xin(r, index, n1)
    else:
        r = f(p, x, prec)
    return r