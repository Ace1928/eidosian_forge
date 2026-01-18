from collections import defaultdict
from sympy.core.numbers import (nan, oo, zoo)
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable
def rational_algorithm(f, x, k, order=4, full=False):
    """
    Rational algorithm for computing
    formula of coefficients of Formal Power Series
    of a function.

    Explanation
    ===========

    Applicable when f(x) or some derivative of f(x)
    is a rational function in x.

    :func:`rational_algorithm` uses :func:`~.apart` function for partial fraction
    decomposition. :func:`~.apart` by default uses 'undetermined coefficients
    method'. By setting ``full=True``, 'Bronstein's algorithm' can be used
    instead.

    Looks for derivative of a function up to 4'th order (by default).
    This can be overridden using order option.

    Parameters
    ==========

    x : Symbol
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    full : bool

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int
    full : bool

    Examples
    ========

    >>> from sympy import log, atan
    >>> from sympy.series.formal import rational_algorithm as ra
    >>> from sympy.abc import x, k

    >>> ra(1 / (1 - x), x, k)
    (1, 0, 0)
    >>> ra(log(1 + x), x, k)
    (-1/((-1)**k*k), 0, 1)

    >>> ra(atan(x), x, k, full=True)
    ((-I/(2*(-I)**k) + I/(2*I**k))/k, 0, 1)

    Notes
    =====

    By setting ``full=True``, range of admissible functions to be solved using
    ``rational_algorithm`` can be increased. This option should be used
    carefully as it can significantly slow down the computation as ``doit`` is
    performed on the :class:`~.RootSum` object returned by the :func:`~.apart`
    function. Use ``full=False`` whenever possible.

    See Also
    ========

    sympy.polys.partfrac.apart

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf

    """
    from sympy.polys import RootSum, apart
    from sympy.integrals import integrate
    diff = f
    ds = []
    for i in range(order + 1):
        if i:
            diff = diff.diff(x)
        if diff.is_rational_function(x):
            coeff, sep = (S.Zero, S.Zero)
            terms = apart(diff, x, full=full)
            if terms.has(RootSum):
                terms = terms.doit()
            for t in Add.make_args(terms):
                num, den = t.as_numer_denom()
                if not den.has(x):
                    sep += t
                else:
                    if isinstance(den, Mul):
                        ind = den.as_independent(x)
                        den = ind[1]
                        num /= ind[0]
                    den, j = den.as_base_exp()
                    a, xterm = den.as_coeff_add(x)
                    if not a:
                        sep += t
                        continue
                    xc = xterm[0].coeff(x)
                    a /= -xc
                    num /= xc ** j
                    ak = (-1) ** j * num * binomial(j + k - 1, k).rewrite(factorial) / a ** (j + k)
                    coeff += ak
            if coeff.is_zero:
                return None
            if coeff.has(x) or coeff.has(zoo) or coeff.has(oo) or coeff.has(nan):
                return None
            for j in range(i):
                coeff = coeff / (k + j + 1)
                sep = integrate(sep, x)
                sep += (ds.pop() - sep).limit(x, 0)
            return (coeff.subs(k, k - i), sep, i)
        else:
            ds.append(diff)
    return None