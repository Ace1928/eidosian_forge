from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_from_dict(d, O):
    """
    Create an sdm from a dictionary.

    Here ``O`` is the monomial order to use.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_from_dict
    >>> from sympy.polys import QQ, lex
    >>> dic = {(1, 1, 0): QQ(1), (1, 0, 0): QQ(2), (0, 1, 0): QQ(0)}
    >>> sdm_from_dict(dic, lex)
    [((1, 1, 0), 1), ((1, 0, 0), 2)]
    """
    return sdm_strip(sdm_sort(list(d.items()), O))