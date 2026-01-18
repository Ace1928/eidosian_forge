from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_to_vector(f, gens, K, n=None):
    """
    Convert sdm ``f`` into a list of polynomial expressions.

    The generators for the polynomial ring are specified via ``gens``. The rank
    of the module is guessed, or passed via ``n``. The ground field is assumed
    to be ``K``.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_to_vector
    >>> from sympy.abc import x, y, z
    >>> from sympy.polys import QQ
    >>> f = [((1, 0, 0, 1), QQ(2)), ((0, 2, 0, 0), QQ(1)), ((0, 0, 2, 0), QQ(1))]
    >>> sdm_to_vector(f, [x, y, z], QQ)
    [x**2 + y**2, 2*z]
    """
    dic = sdm_to_dict(f)
    dics = {}
    for k, v in dic.items():
        dics.setdefault(k[0], []).append((k[1:], v))
    n = n or len(dics)
    res = []
    for k in range(n):
        if k in dics:
            res.append(Poly(dict(dics[k]), gens=gens, domain=K).as_expr())
        else:
            res.append(S.Zero)
    return res