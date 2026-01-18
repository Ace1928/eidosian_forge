from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_from_raw_dict(f, K):
    """
    Create a ``K[x]`` polynomial from a raw ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_raw_dict

    >>> dup_from_raw_dict({0: ZZ(7), 2: ZZ(5), 4: ZZ(1)}, ZZ)
    [1, 0, 5, 0, 7]

    """
    if not f:
        return []
    n, h = (max(f.keys()), [])
    for k in range(n, -1, -1):
        h.append(f.get(k, K.zero))
    return dup_strip(h)