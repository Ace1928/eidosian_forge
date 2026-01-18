from the names used in Bronstein's book.
from types import GeneratorType
from functools import reduce
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, cos,
from .integrals import integrate, Integral
from .heurisch import _symbols
from sympy.polys.polyerrors import DomainError, PolynomialError
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,
from sympy.polys.rootoftools import RootSum
from sympy.utilities.iterables import numbered_symbols
def integer_powers(exprs):
    """
    Rewrites a list of expressions as integer multiples of each other.

    Explanation
    ===========

    For example, if you have [x, x/2, x**2 + 1, 2*x/3], then you can rewrite
    this as [(x/6) * 6, (x/6) * 3, (x**2 + 1) * 1, (x/6) * 4]. This is useful
    in the Risch integration algorithm, where we must write exp(x) + exp(x/2)
    as (exp(x/2))**2 + exp(x/2), but not as exp(x) + sqrt(exp(x)) (this is
    because only the transcendental case is implemented and we therefore cannot
    integrate algebraic extensions). The integer multiples returned by this
    function for each term are the smallest possible (their content equals 1).

    Returns a list of tuples where the first element is the base term and the
    second element is a list of `(item, factor)` terms, where `factor` is the
    integer multiplicative factor that must multiply the base term to obtain
    the original item.

    The easiest way to understand this is to look at an example:

    >>> from sympy.abc import x
    >>> from sympy.integrals.risch import integer_powers
    >>> integer_powers([x, x/2, x**2 + 1, 2*x/3])
    [(x/6, [(x, 6), (x/2, 3), (2*x/3, 4)]), (x**2 + 1, [(x**2 + 1, 1)])]

    We can see how this relates to the example at the beginning of the
    docstring.  It chose x/6 as the first base term.  Then, x can be written as
    (x/2) * 2, so we get (0, 2), and so on. Now only element (x**2 + 1)
    remains, and there are no other terms that can be written as a rational
    multiple of that, so we get that it can be written as (x**2 + 1) * 1.

    """
    terms = {}
    for term in exprs:
        for trm, trm_list in terms.items():
            a = cancel(term / trm)
            if a.is_Rational:
                trm_list.append((term, a))
                break
        else:
            terms[term] = [(term, S.One)]
    newterms = {}
    for term, term_list in terms.items():
        common_denom = reduce(ilcm, [i.as_numer_denom()[1] for _, i in term_list])
        newterm = term / common_denom
        newmults = [(i, j * common_denom) for i, j in term_list]
        newterms[newterm] = newmults
    return sorted(iter(newterms.items()), key=lambda item: item[0].sort_key())