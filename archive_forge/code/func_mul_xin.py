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
def mul_xin(p, i, n):
    """
    Return `p*x_i**n`.

    `x\\_i` is the ith variable in ``p``.
    """
    R = p.ring
    q = R(0)
    for k, v in p.items():
        k1 = list(k)
        k1[i] += n
        q[tuple(k1)] = v
    return q