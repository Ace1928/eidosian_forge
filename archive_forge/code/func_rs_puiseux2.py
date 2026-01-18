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
def rs_puiseux2(f, p, q, x, prec):
    """
    Return the puiseux series for `f(p, q, x, prec)`.

    To be used when function ``f`` is implemented only for regular series.
    """
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = n * den // igcd(n, den)
        elif power != int(power):
            den = power.denominator
            n = n * den // igcd(n, den)
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, q, x, prec * n)
        n1 = QQ(1, n)
        r = pow_xin(r, index, n1)
    else:
        r = f(p, q, x, prec)
    return r