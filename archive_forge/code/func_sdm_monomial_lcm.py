from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_monomial_lcm(A, B):
    """
    Return the "least common multiple" of ``A`` and ``B``.

    IF `A = M e_j` and `B = N e_j`, where `M` and `N` are polynomial monomials,
    this returns `\\lcm(M, N) e_j`. Note that ``A`` and ``B`` involve distinct
    monomials.

    Otherwise the result is undefined.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_monomial_lcm
    >>> sdm_monomial_lcm((1, 2, 3), (1, 0, 5))
    (1, 2, 5)
    """
    return (A[0],) + monomial_lcm(A[1:], B[1:])