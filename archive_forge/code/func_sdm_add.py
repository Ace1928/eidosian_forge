from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_add(f, g, O, K):
    """
    Add two module elements ``f``, ``g``.

    Addition is done over the ground field ``K``, monomials are ordered
    according to ``O``.

    Examples
    ========

    All examples use lexicographic order.

    `(xy f_1) + (f_2) = f_2 + xy f_1`

    >>> from sympy.polys.distributedmodules import sdm_add
    >>> from sympy.polys import lex, QQ
    >>> sdm_add([((1, 1, 1), QQ(1))], [((2, 0, 0), QQ(1))], lex, QQ)
    [((2, 0, 0), 1), ((1, 1, 1), 1)]

    `(xy f_1) + (-xy f_1)` = 0`

    >>> sdm_add([((1, 1, 1), QQ(1))], [((1, 1, 1), QQ(-1))], lex, QQ)
    []

    `(f_1) + (2f_1) = 3f_1`

    >>> sdm_add([((1, 0, 0), QQ(1))], [((1, 0, 0), QQ(2))], lex, QQ)
    [((1, 0, 0), 3)]

    `(yf_1) + (xf_1) = xf_1 + yf_1`

    >>> sdm_add([((1, 0, 1), QQ(1))], [((1, 1, 0), QQ(1))], lex, QQ)
    [((1, 1, 0), 1), ((1, 0, 1), 1)]
    """
    h = dict(f)
    for monom, c in g:
        if monom in h:
            coeff = h[monom] + c
            if not coeff:
                del h[monom]
            else:
                h[monom] = coeff
        else:
            h[monom] = c
    return sdm_from_dict(h, O)