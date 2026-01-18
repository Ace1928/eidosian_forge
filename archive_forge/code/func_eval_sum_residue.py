from typing import Tuple as tTuple
from sympy.calculus.singularities import is_decreasing
from sympy.calculus.accumulationbounds import AccumulationBounds
from .expr_with_intlimits import ExprWithIntLimits
from .expr_with_limits import AddWithLimits
from .gosper import gosper_sum
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Float, _illegal
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Wild, Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cot, csc
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import PolynomialError, PolificationFailed
from sympy.polys.polytools import parallel_poly_from_expr, Poly, factor
from sympy.polys.rationaltools import together
from sympy.series.limitseq import limit_seq
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.sets.sets import FiniteSet, Interval
from sympy.utilities.iterables import sift
import itertools
def eval_sum_residue(f, i_a_b):
    """Compute the infinite summation with residues

    Notes
    =====

    If $f(n), g(n)$ are polynomials with $\\deg(g(n)) - \\deg(f(n)) \\ge 2$,
    some infinite summations can be computed by the following residue
    evaluations.

    .. math::
        \\sum_{n=-\\infty, g(n) \\ne 0}^{\\infty} \\frac{f(n)}{g(n)} =
        -\\pi \\sum_{\\alpha|g(\\alpha)=0}
        \\text{Res}(\\cot(\\pi x) \\frac{f(x)}{g(x)}, \\alpha)

    .. math::
        \\sum_{n=-\\infty, g(n) \\ne 0}^{\\infty} (-1)^n \\frac{f(n)}{g(n)} =
        -\\pi \\sum_{\\alpha|g(\\alpha)=0}
        \\text{Res}(\\csc(\\pi x) \\frac{f(x)}{g(x)}, \\alpha)

    Examples
    ========

    >>> from sympy import Sum, oo, Symbol
    >>> x = Symbol('x')

    Doubly infinite series of rational functions.

    >>> Sum(1 / (x**2 + 1), (x, -oo, oo)).doit()
    pi/tanh(pi)

    Doubly infinite alternating series of rational functions.

    >>> Sum((-1)**x / (x**2 + 1), (x, -oo, oo)).doit()
    pi/sinh(pi)

    Infinite series of even rational functions.

    >>> Sum(1 / (x**2 + 1), (x, 0, oo)).doit()
    1/2 + pi/(2*tanh(pi))

    Infinite series of alternating even rational functions.

    >>> Sum((-1)**x / (x**2 + 1), (x, 0, oo)).doit()
    pi/(2*sinh(pi)) + 1/2

    This also have heuristics to transform arbitrarily shifted summand or
    arbitrarily shifted summation range to the canonical problem the
    formula can handle.

    >>> Sum(1 / (x**2 + 2*x + 2), (x, -1, oo)).doit()
    1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 4*x + 5), (x, -2, oo)).doit()
    1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 1), (x, 1, oo)).doit()
    -1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 1), (x, 2, oo)).doit()
    -1 + pi/(2*tanh(pi))

    References
    ==========

    .. [#] http://www.supermath.info/InfiniteSeriesandtheResidueTheorem.pdf

    .. [#] Asmar N.H., Grafakos L. (2018) Residue Theory.
           In: Complex Analysis with Applications.
           Undergraduate Texts in Mathematics. Springer, Cham.
           https://doi.org/10.1007/978-3-319-94063-2_5
    """
    i, a, b = i_a_b

    def is_even_function(numer, denom):
        """Test if the rational function is an even function"""
        numer_even = all((i % 2 == 0 for i, in numer.monoms()))
        denom_even = all((i % 2 == 0 for i, in denom.monoms()))
        numer_odd = all((i % 2 == 1 for i, in numer.monoms()))
        denom_odd = all((i % 2 == 1 for i, in denom.monoms()))
        return numer_even and denom_even or (numer_odd and denom_odd)

    def match_rational(f, i):
        numer, denom = f.as_numer_denom()
        try:
            (numer, denom), opt = parallel_poly_from_expr((numer, denom), i)
        except (PolificationFailed, PolynomialError):
            return None
        return (numer, denom)

    def get_poles(denom):
        roots = denom.sqf_part().all_roots()
        roots = sift(roots, lambda x: x.is_integer)
        if None in roots:
            return None
        int_roots, nonint_roots = (roots[True], roots[False])
        return (int_roots, nonint_roots)

    def get_shift(denom):
        n = denom.degree(i)
        a = denom.coeff_monomial(i ** n)
        b = denom.coeff_monomial(i ** (n - 1))
        shift = -b / a / n
        return shift
    z = Dummy('z')

    def get_residue_factor(numer, denom, alternating):
        residue_factor = (numer.as_expr() / denom.as_expr()).subs(i, z)
        if not alternating:
            residue_factor *= cot(S.Pi * z)
        else:
            residue_factor *= csc(S.Pi * z)
        return residue_factor
    if f.free_symbols - {i}:
        return None
    if not (a.is_Integer or a in (S.Infinity, S.NegativeInfinity)):
        return None
    if not (b.is_Integer or b in (S.Infinity, S.NegativeInfinity)):
        return None
    if a != S.NegativeInfinity and b != S.Infinity:
        return None
    match = match_rational(f, i)
    if match:
        alternating = False
        numer, denom = match
    else:
        match = match_rational(f / S.NegativeOne ** i, i)
        if match:
            alternating = True
            numer, denom = match
        else:
            return None
    if denom.degree(i) - numer.degree(i) < 2:
        return None
    if (a, b) == (S.NegativeInfinity, S.Infinity):
        poles = get_poles(denom)
        if poles is None:
            return None
        int_roots, nonint_roots = poles
        if int_roots:
            return None
        residue_factor = get_residue_factor(numer, denom, alternating)
        residues = [residue(residue_factor, z, root) for root in nonint_roots]
        return -S.Pi * sum(residues)
    if not (a.is_finite and b is S.Infinity):
        return None
    if not is_even_function(numer, denom):
        shift = get_shift(denom)
        if not shift.is_Integer:
            return None
        if shift == 0:
            return None
        numer = numer.shift(shift)
        denom = denom.shift(shift)
        if not is_even_function(numer, denom):
            return None
        if alternating:
            f = S.NegativeOne ** i * (S.NegativeOne ** shift * numer.as_expr() / denom.as_expr())
        else:
            f = numer.as_expr() / denom.as_expr()
        return eval_sum_residue(f, (i, a - shift, b - shift))
    poles = get_poles(denom)
    if poles is None:
        return None
    int_roots, nonint_roots = poles
    if int_roots:
        int_roots = [int(root) for root in int_roots]
        int_roots_max = max(int_roots)
        int_roots_min = min(int_roots)
        if not len(int_roots) == int_roots_max - int_roots_min + 1:
            return None
        if a <= max(int_roots):
            return None
    residue_factor = get_residue_factor(numer, denom, alternating)
    residues = [residue(residue_factor, z, root) for root in int_roots + nonint_roots]
    full_sum = -S.Pi * sum(residues)
    if not int_roots:
        half_sum = (full_sum + f.xreplace({i: 0})) / 2
        extraneous_neg = [f.xreplace({i: i0}) for i0 in range(int(a), 0)]
        extraneous_pos = [f.xreplace({i: i0}) for i0 in range(0, int(a))]
        result = half_sum + sum(extraneous_neg) - sum(extraneous_pos)
        return result
    half_sum = full_sum / 2
    extraneous = [f.xreplace({i: i0}) for i0 in range(max(int_roots) + 1, int(a))]
    result = half_sum - sum(extraneous)
    return result