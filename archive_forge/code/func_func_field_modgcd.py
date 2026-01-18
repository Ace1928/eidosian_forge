from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def func_field_modgcd(f, g):
    """
    Compute the GCD of two polynomials `f` and `g` in
    `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]` using a modular algorithm.

    The algorithm first computes the primitive associate
    `\\check m_{\\alpha}(z)` of the minimal polynomial `m_{\\alpha}` in
    `\\mathbb{Z}[z]` and the primitive associates of `f` and `g` in
    `\\mathbb{Z}[x_1, \\ldots, x_{n-1}][z]/(\\check m_{\\alpha})[x_0]`. Then it
    computes the GCD in
    `\\mathbb Q(x_1, \\ldots, x_{n-1})[z]/(m_{\\alpha}(z))[x_0]`.
    This is done by calculating the GCD in
    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem and Rational Reconstuction. The GCD over
    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]` is
    computed with a recursive subroutine, which evaluates the polynomials at
    `x_{n-1} = a` for suitable evaluation points `a \\in \\mathbb Z_p` and
    then calls itself recursively until the ground domain does no longer
    contain any parameters. For
    `\\mathbb{Z}_p[z]/(\\check m_{\\alpha}(z))[x_0]` the Euclidean Algorithm is
    used. The results of those recursive calls are then interpolated and
    Rational Function Reconstruction is used to obtain the correct
    coefficients. The results, both in
    `\\mathbb Q(x_1, \\ldots, x_{n-1})[z]/(m_{\\alpha}(z))[x_0]` and
    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]`, are
    verified by a fraction free trial division.

    Apart from the above GCD computation some GCDs in
    `\\mathbb Q(\\alpha)[x_1, \\ldots, x_{n-1}]` have to be calculated,
    because treating the polynomials as univariate ones can result in
    a spurious content of the GCD. For this ``func_field_modgcd`` is
    called recursively.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`

    Returns
    =======

    h : PolyElement
        monic GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\\frac f h`
    cfg : PolyElement
        cofactor of `g`, i.e. `\\frac g h`

    Examples
    ========

    >>> from sympy.polys.modulargcd import func_field_modgcd
    >>> from sympy.polys import AlgebraicField, QQ, ring
    >>> from sympy import sqrt

    >>> A = AlgebraicField(QQ, sqrt(2))
    >>> R, x = ring('x', A)

    >>> f = x**2 - 2
    >>> g = x + sqrt(2)

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == x + sqrt(2)
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> R, x, y = ring('x, y', A)

    >>> f = x**2 + 2*sqrt(2)*x*y + 2*y**2
    >>> g = x + sqrt(2)*y

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == x + sqrt(2)*y
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = x + sqrt(2)*y
    >>> g = x + y

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == R.one
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Hoeij04]_

    """
    ring = f.ring
    domain = ring.domain
    n = ring.ngens
    assert ring == g.ring and domain.is_Algebraic
    result = _trivial_gcd(f, g)
    if result is not None:
        return result
    z = Dummy('z')
    ZZring = ring.clone(symbols=ring.symbols + (z,), domain=domain.domain.get_ring())
    if n == 1:
        f_ = _to_ZZ_poly(f, ZZring)
        g_ = _to_ZZ_poly(g, ZZring)
        minpoly = ZZring.drop(0).from_dense(domain.mod.rep)
        h = _func_field_modgcd_m(f_, g_, minpoly)
        h = _to_ANP_poly(h, ring)
    else:
        contx0f, f = _primitive_in_x0(f)
        contx0g, g = _primitive_in_x0(g)
        contx0h = func_field_modgcd(contx0f, contx0g)[0]
        ZZring_ = ZZring.drop_to_ground(*range(1, n))
        f_ = _to_ZZ_poly(f, ZZring_)
        g_ = _to_ZZ_poly(g, ZZring_)
        minpoly = _minpoly_from_dense(domain.mod, ZZring_.drop(0))
        h = _func_field_modgcd_m(f_, g_, minpoly)
        h = _to_ANP_poly(h, ring)
        contx0h_, h = _primitive_in_x0(h)
        h *= contx0h.set_ring(ring)
        f *= contx0f.set_ring(ring)
        g *= contx0g.set_ring(ring)
    h = h.quo_ground(h.LC)
    return (h, f.quo(h), g.quo(h))