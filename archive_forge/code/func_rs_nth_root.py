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
def rs_nth_root(p, n, x, prec):
    """
    Multivariate series expansion of the nth root of ``p``.

    Parameters
    ==========

    p : Expr
        The polynomial to computer the root of.
    n : integer
        The order of the root to be computed.
    x : :class:`~.PolyElement`
    prec : integer
        Order of the expanded series.

    Notes
    =====

    The result of this function is dependent on the ring over which the
    polynomial has been defined. If the answer involves a root of a constant,
    make sure that the polynomial is over a real field. It cannot yet handle
    roots of symbols.

    Examples
    ========

    >>> from sympy.polys.domains import QQ, RR
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_nth_root
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_nth_root(1 + x + x*y, -3, x, 3)
    2/9*x**2*y**2 + 4/9*x**2*y + 2/9*x**2 - 1/3*x*y - 1/3*x + 1
    >>> R, x, y = ring('x, y', RR)
    >>> rs_nth_root(3 + x + x*y, 3, x, 2)
    0.160249952256379*x*y + 0.160249952256379*x + 1.44224957030741
    """
    if n == 0:
        if p == 0:
            raise ValueError('0**0 expression')
        else:
            return p.ring(1)
    if n == 1:
        return rs_trunc(p, x, prec)
    R = p.ring
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    p = mul_xin(p, index, -m)
    prec -= m
    if _has_constant_term(p - 1, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = c_expr ** QQ(1, n)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(c_expr ** QQ(1, n))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        else:
            try:
                const = R(c ** Rational(1, n))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        res = rs_nth_root(p / c, n, x, prec) * const
    else:
        res = _nth_root1(p, n, x, prec)
    if m:
        m = QQ(m, n)
        res = mul_xin(res, index, m)
    return res