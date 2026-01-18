from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_swap(f, i, j, u, K):
    """
    Transform ``K[..x_i..x_j..]`` to ``K[..x_j..x_i..]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_swap

    >>> f = ZZ.map([[[2], [1, 0]], []])

    >>> dmp_swap(f, 0, 1, 2, ZZ)
    [[[2], []], [[1, 0], []]]
    >>> dmp_swap(f, 1, 2, 2, ZZ)
    [[[1], [2, 0]], [[]]]
    >>> dmp_swap(f, 0, 2, 2, ZZ)
    [[[1, 0]], [[2, 0], []]]

    """
    if i < 0 or j < 0 or i > u or (j > u):
        raise IndexError('0 <= i < j <= %s expected' % u)
    elif i == j:
        return f
    F, H = (dmp_to_dict(f, u), {})
    for exp, coeff in F.items():
        H[exp[:i] + (exp[j],) + exp[i + 1:j] + (exp[i],) + exp[j + 1:]] = coeff
    return dmp_from_dict(H, u, K)