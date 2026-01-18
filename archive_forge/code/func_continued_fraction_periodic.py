from __future__ import annotations
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.utilities.misc import as_int
def continued_fraction_periodic(p, q, d=0, s=1) -> list:
    """
    Find the periodic continued fraction expansion of a quadratic irrational.

    Compute the continued fraction expansion of a rational or a
    quadratic irrational number, i.e. `\\frac{p + s\\sqrt{d}}{q}`, where
    `p`, `q \\ne 0` and `d \\ge 0` are integers.

    Returns the continued fraction representation (canonical form) as
    a list of integers, optionally ending (for quadratic irrationals)
    with list of integers representing the repeating digits.

    Parameters
    ==========

    p : int
        the rational part of the number's numerator
    q : int
        the denominator of the number
    d : int, optional
        the irrational part (discriminator) of the number's numerator
    s : int, optional
        the coefficient of the irrational part

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic
    >>> continued_fraction_periodic(3, 2, 7)
    [2, [1, 4, 1, 1]]

    Golden ratio has the simplest continued fraction expansion:

    >>> continued_fraction_periodic(1, 2, 5)
    [[1]]

    If the discriminator is zero or a perfect square then the number will be a
    rational number:

    >>> continued_fraction_periodic(4, 3, 0)
    [1, 3]
    >>> continued_fraction_periodic(4, 3, 49)
    [3, 1, 2]

    See Also
    ========

    continued_fraction_iterator, continued_fraction_reduce

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Periodic_continued_fraction
    .. [2] K. Rosen. Elementary Number theory and its applications.
           Addison-Wesley, 3 Sub edition, pages 379-381, January 1992.

    """
    from sympy.functions import sqrt, floor
    p, q, d, s = list(map(as_int, [p, q, d, s]))
    if d < 0:
        raise ValueError('expected non-negative for `d` but got %s' % d)
    if q == 0:
        raise ValueError('The denominator cannot be 0.')
    if not s:
        d = 0
    sd = sqrt(d)
    if sd.is_Integer:
        return list(continued_fraction_iterator(Rational(p + s * sd, q)))
    if q < 0:
        p, q, s = (-p, -q, -s)
    n = (p + s * sd) / q
    if n < 0:
        w = floor(-n)
        f = -n - w
        one_f = continued_fraction(1 - f)
        one_f[0] -= w + 1
        return one_f
    d *= s ** 2
    sd *= s
    if (d - p ** 2) % q:
        d *= q ** 2
        sd *= q
        p *= q
        q *= q
    terms: list[int] = []
    pq = {}
    while (p, q) not in pq:
        pq[p, q] = len(terms)
        terms.append((p + sd) // q)
        p = terms[-1] * q - p
        q = (d - p ** 2) // q
    i = pq[p, q]
    return terms[:i] + [terms[i:]]