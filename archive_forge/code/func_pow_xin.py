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
def pow_xin(p, i, n):
    """
    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import pow_xin
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x**QQ(2,5) + x + x**QQ(2,3)
    >>> index = p.ring.gens.index(x)
    >>> pow_xin(p, index, 15)
    x**15 + x**10 + x**6
    """
    R = p.ring
    q = R(0)
    for k, v in p.items():
        k1 = list(k)
        k1[i] *= n
        q[tuple(k1)] = v
    return q