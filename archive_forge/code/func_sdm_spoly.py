from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_spoly(f, g, O, K, phantom=None):
    """
    Compute the generalized s-polynomial of ``f`` and ``g``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    This is invalid if either of ``f`` or ``g`` is zero.

    If the leading terms of `f` and `g` involve different basis elements of
    `F`, their s-poly is defined to be zero. Otherwise it is a certain linear
    combination of `f` and `g` in which the leading terms cancel.
    See [SCA, defn 2.3.6] for details.

    If ``phantom`` is not ``None``, it should be a pair of module elements on
    which to perform the same operation(s) as on ``f`` and ``g``. The in this
    case both results are returned.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_spoly
    >>> from sympy.polys import QQ, lex
    >>> f = [((2, 1, 1), QQ(1)), ((1, 0, 1), QQ(1))]
    >>> g = [((2, 3, 0), QQ(1))]
    >>> h = [((1, 2, 3), QQ(1))]
    >>> sdm_spoly(f, h, lex, QQ)
    []
    >>> sdm_spoly(f, g, lex, QQ)
    [((1, 2, 1), 1)]
    """
    if not f or not g:
        return sdm_zero()
    LM1 = sdm_LM(f)
    LM2 = sdm_LM(g)
    if LM1[0] != LM2[0]:
        return sdm_zero()
    LM1 = LM1[1:]
    LM2 = LM2[1:]
    lcm = monomial_lcm(LM1, LM2)
    m1 = monomial_div(lcm, LM1)
    m2 = monomial_div(lcm, LM2)
    c = K.quo(-sdm_LC(f, K), sdm_LC(g, K))
    r1 = sdm_add(sdm_mul_term(f, (m1, K.one), O, K), sdm_mul_term(g, (m2, c), O, K), O, K)
    if phantom is None:
        return r1
    r2 = sdm_add(sdm_mul_term(phantom[0], (m1, K.one), O, K), sdm_mul_term(phantom[1], (m2, c), O, K), O, K)
    return (r1, r2)