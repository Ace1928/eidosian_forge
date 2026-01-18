from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_from_dict(f, K):
    """
    Create a ``K[x]`` polynomial from a ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_dict

    >>> dup_from_dict({(0,): ZZ(7), (2,): ZZ(5), (4,): ZZ(1)}, ZZ)
    [1, 0, 5, 0, 7]
    >>> dup_from_dict({}, ZZ)
    []

    """
    if not f:
        return []
    n, h = (max(f.keys()), [])
    if isinstance(n, int):
        for k in range(n, -1, -1):
            h.append(f.get(k, K.zero))
    else:
        n, = n
        for k in range(n, -1, -1):
            h.append(f.get((k,), K.zero))
    return dup_strip(h)