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
def rs_series_reversion(p, x, n, y):
    """
    Reversion of a series.

    ``p`` is a series with ``O(x**n)`` of the form $p = ax + f(x)$
    where $a$ is a number different from 0.

    $f(x) = \\sum_{k=2}^{n-1} a_kx_k$

    Parameters
    ==========

      a_k : Can depend polynomially on other variables, not indicated.
      x : Variable with name x.
      y : Variable with name y.

    Returns
    =======

    Solve $p = y$, that is, given $ax + f(x) - y = 0$,
    find the solution $x = r(y)$ up to $O(y^n)$.

    Algorithm
    =========

    If $r_i$ is the solution at order $i$, then:
    $ar_i + f(r_i) - y = O\\left(y^{i + 1}\\right)$

    and if $r_{i + 1}$ is the solution at order $i + 1$, then:
    $ar_{i + 1} + f(r_{i + 1}) - y = O\\left(y^{i + 2}\\right)$

    We have, $r_{i + 1} = r_i + e$, such that,
    $ae + f(r_i) = O\\left(y^{i + 2}\\right)$
    or $e = -f(r_i)/a$

    So we use the recursion relation:
    $r_{i + 1} = r_i - f(r_i)/a$
    with the boundary condition: $r_1 = y$

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_reversion, rs_trunc
    >>> R, x, y, a, b = ring('x, y, a, b', QQ)
    >>> p = x - x**2 - 2*b*x**2 + 2*a*b*x**2
    >>> p1 = rs_series_reversion(p, x, 3, y); p1
    -2*y**2*a*b + 2*y**2*b + y**2 + y
    >>> rs_trunc(p.compose(x, p1), y, 3)
    y
    """
    if rs_is_puiseux(p, x):
        raise NotImplementedError
    R = p.ring
    nx = R.gens.index(x)
    y = R(y)
    ny = R.gens.index(y)
    if _has_constant_term(p, x):
        raise ValueError('p must not contain a constant term in the series variable')
    a = _coefficient_t(p, (nx, 1))
    zm = R.zero_monom
    assert zm in a and len(a) == 1
    a = a[zm]
    r = y / a
    for i in range(2, n):
        sp = rs_subs(p, {x: r}, y, i + 1)
        sp = _coefficient_t(sp, (ny, i)) * y ** i
        r -= sp / a
    return r