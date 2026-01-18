from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_apply_pairs(f, g, h, args, K):
    """
    Apply ``h`` to pairs of coefficients of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_apply_pairs

    >>> h = lambda x, y, z: 2*x + y - z

    >>> dup_apply_pairs([1, 2, 3], [3, 2, 1], h, (1,), ZZ)
    [4, 5, 6]

    """
    n, m = (len(f), len(g))
    if n != m:
        if n > m:
            g = [K.zero] * (n - m) + g
        else:
            f = [K.zero] * (m - n) + f
    result = []
    for a, b in zip(f, g):
        result.append(h(a, b, *args))
    return dup_strip(result)