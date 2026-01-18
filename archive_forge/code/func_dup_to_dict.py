from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_to_dict(f, K=None, zero=False):
    """
    Convert ``K[x]`` polynomial to a ``dict``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_to_dict

    >>> dup_to_dict([1, 0, 5, 0, 7])
    {(0,): 7, (2,): 5, (4,): 1}
    >>> dup_to_dict([])
    {}

    """
    if not f and zero:
        return {(0,): K.zero}
    n, result = (len(f) - 1, {})
    for k in range(0, n + 1):
        if f[n - k]:
            result[k,] = f[n - k]
    return result