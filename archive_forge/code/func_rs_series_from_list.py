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
def rs_series_from_list(p, c, x, prec, concur=1):
    """
    Return a series `sum c[n]*p**n` modulo `O(x**prec)`.

    It reduces the number of multiplications by summing concurrently.

    `ax = [1, p, p**2, .., p**(J - 1)]`
    `s = sum(c[i]*ax[i]` for i in `range(r, (r + 1)*J))*p**((K - 1)*J)`
    with `K >= (n + 1)/J`

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_from_list, rs_trunc
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + x + 1
    >>> c = [1, 2, 3]
    >>> rs_series_from_list(p, c, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> rs_trunc(1 + 2*p + 3*p**2, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> pc = R.from_list(list(reversed(c)))
    >>> rs_trunc(pc.compose(x, p), x, 4)
    6*x**3 + 11*x**2 + 8*x + 6

    """
    '\n    See Also\n    ========\n\n    sympy.polys.rings.PolyRing.compose\n\n    '
    R = p.ring
    n = len(c)
    if not concur:
        q = R(1)
        s = c[0] * q
        for i in range(1, n):
            q = rs_mul(q, p, x, prec)
            s += c[i] * q
        return s
    J = int(math.sqrt(n) + 1)
    K, r = divmod(n, J)
    if r:
        K += 1
    ax = [R(1)]
    q = R(1)
    if len(p) < 20:
        for i in range(1, J):
            q = rs_mul(q, p, x, prec)
            ax.append(q)
    else:
        for i in range(1, J):
            if i % 2 == 0:
                q = rs_square(ax[i // 2], x, prec)
            else:
                q = rs_mul(q, p, x, prec)
            ax.append(q)
    pj = rs_mul(ax[-1], p, x, prec)
    b = R(1)
    s = R(0)
    for k in range(K - 1):
        r = J * k
        s1 = c[r]
        for j in range(1, J):
            s1 += c[r + j] * ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
        b = rs_mul(b, pj, x, prec)
        if not b:
            break
    k = K - 1
    r = J * k
    if r < n:
        s1 = c[r] * R(1)
        for j in range(1, J):
            if r + j >= n:
                break
            s1 += c[r + j] * ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
    return s