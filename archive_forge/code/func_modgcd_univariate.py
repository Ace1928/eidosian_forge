from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def modgcd_univariate(f, g):
    """
    Computes the GCD of two polynomials in `\\mathbb{Z}[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two univariate integer polynomials
    `f` and `g` by computing the GCD in `\\mathbb{Z}_p[x]` for suitable
    primes `p` and then reconstructing the coefficients with the Chinese
    Remainder Theorem. Trial division is only made for candidates which
    are very likely the desired GCD.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)

    >>> f = x**5 - 1
    >>> g = x - 1

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (x - 1, x**4 + x**3 + x**2 + x + 1, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = 6*x**2 - 6
    >>> g = 2*x**2 + 4*x + 2

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (2*x + 2, 3*x - 3, x + 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ
    result = _trivial_gcd(f, g)
    if result is not None:
        return result
    ring = f.ring
    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)
    bound = _degree_bound_univariate(f, g)
    if bound == 0:
        return (ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch))
    gamma = ring.domain.gcd(f.LC, g.LC)
    m = 1
    p = 1
    while True:
        p = nextprime(p)
        while gamma % p == 0:
            p = nextprime(p)
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        hp = _gf_gcd(fp, gp, p)
        deghp = hp.degree()
        if deghp > bound:
            continue
        elif deghp < bound:
            m = 1
            bound = deghp
            continue
        hp = hp.mul_ground(gamma).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue
        hm = _chinese_remainder_reconstruction_univariate(hp, hlastm, p, m)
        m *= p
        if not hm == hlastm:
            hlastm = hm
            continue
        h = hm.quo_ground(hm.content())
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and (not grem):
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return (h, cff, cfg)