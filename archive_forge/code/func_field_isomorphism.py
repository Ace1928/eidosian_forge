from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public
from mpmath import MPContext
@public
def field_isomorphism(a, b, *, fast=True):
    """
    Find an embedding of one number field into another.

    Explanation
    ===========

    This function looks for an isomorphism from $\\mathbb{Q}(a)$ onto some
    subfield of $\\mathbb{Q}(b)$. Thus, it solves the Subfield Problem.

    Examples
    ========

    >>> from sympy import sqrt, field_isomorphism, I
    >>> print(field_isomorphism(3, sqrt(2)))  # doctest: +SKIP
    [3]
    >>> print(field_isomorphism( I*sqrt(3), I*sqrt(3)/2))  # doctest: +SKIP
    [2, 0]

    Parameters
    ==========

    a : :py:class:`~.Expr`
        Any expression representing an algebraic number.
    b : :py:class:`~.Expr`
        Any expression representing an algebraic number.
    fast : boolean, optional (default=True)
        If ``True``, we first attempt a potentially faster way of computing the
        isomorphism, falling back on a slower method if this fails. If
        ``False``, we go directly to the slower method, which is guaranteed to
        return a result.

    Returns
    =======

    List of rational numbers, or None
        If $\\mathbb{Q}(a)$ is not isomorphic to some subfield of
        $\\mathbb{Q}(b)$, then return ``None``. Otherwise, return a list of
        rational numbers representing an element of $\\mathbb{Q}(b)$ to which
        $a$ may be mapped, in order to define a monomorphism, i.e. an
        isomorphism from $\\mathbb{Q}(a)$ to some subfield of $\\mathbb{Q}(b)$.
        The elements of the list are the coefficients of falling powers of $b$.

    """
    a, b = (sympify(a), sympify(b))
    if not a.is_AlgebraicNumber:
        a = AlgebraicNumber(a)
    if not b.is_AlgebraicNumber:
        b = AlgebraicNumber(b)
    a = a.to_primitive_element()
    b = b.to_primitive_element()
    if a == b:
        return a.coeffs()
    n = a.minpoly.degree()
    m = b.minpoly.degree()
    if n == 1:
        return [a.root]
    if m % n != 0:
        return None
    if fast:
        try:
            result = field_isomorphism_pslq(a, b)
            if result is not None:
                return result
        except NotImplementedError:
            pass
    return field_isomorphism_factor(a, b)