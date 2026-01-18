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
def rs_is_puiseux(p, x):
    """
    Test if ``p`` is Puiseux series in ``x``.

    Raise an exception if it has a negative power in ``x``.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_is_puiseux
    >>> R, x = ring('x', QQ)
    >>> p = x**QQ(2,5) + x**QQ(2,3) + x
    >>> rs_is_puiseux(p, x)
    True
    """
    index = p.ring.gens.index(x)
    for k in p:
        if k[index] != int(k[index]):
            return True
        if k[index] < 0:
            raise ValueError('The series is not regular in %s' % x)
    return False