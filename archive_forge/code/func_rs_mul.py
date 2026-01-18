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
def rs_mul(p1, p2, x, prec):
    """
    Return the product of the given two series, modulo ``O(x**prec)``.

    ``x`` is the series variable or its position in the generators.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_mul
    >>> R, x = ring('x', QQ)
    >>> p1 = x**2 + 2*x + 1
    >>> p2 = x + 1
    >>> rs_mul(p1, p2, x, 3)
    3*x**2 + 3*x + 1
    """
    R = p1.ring
    p = R.zero
    if R.__class__ != p2.ring.__class__ or R != p2.ring:
        raise ValueError('p1 and p2 must have the same ring')
    iv = R.gens.index(x)
    if not isinstance(p2, PolyElement):
        raise ValueError('p2 must be a polynomial')
    if R == p2.ring:
        get = p.get
        items2 = list(p2.items())
        items2.sort(key=lambda e: e[0][iv])
        if R.ngens == 1:
            for exp1, v1 in p1.items():
                for exp2, v2 in items2:
                    exp = exp1[0] + exp2[0]
                    if exp < prec:
                        exp = (exp,)
                        p[exp] = get(exp, 0) + v1 * v2
                    else:
                        break
        else:
            monomial_mul = R.monomial_mul
            for exp1, v1 in p1.items():
                for exp2, v2 in items2:
                    if exp1[iv] + exp2[iv] < prec:
                        exp = monomial_mul(exp1, exp2)
                        p[exp] = get(exp, 0) + v1 * v2
                    else:
                        break
    p.strip_zero()
    return p