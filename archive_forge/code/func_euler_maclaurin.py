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
def euler_maclaurin(self, m=0, n=0, eps=0, eval_integral=True):
    """
        Return an Euler-Maclaurin approximation of self, where m is the
        number of leading terms to sum directly and n is the number of
        terms in the tail.

        With m = n = 0, this is simply the corresponding integral
        plus a first-order endpoint correction.

        Returns (s, e) where s is the Euler-Maclaurin approximation
        and e is the estimated error (taken to be the magnitude of
        the first omitted term in the tail):

            >>> from sympy.abc import k, a, b
            >>> from sympy import Sum
            >>> Sum(1/k, (k, 2, 5)).doit().evalf()
            1.28333333333333
            >>> s, e = Sum(1/k, (k, 2, 5)).euler_maclaurin()
            >>> s
            -log(2) + 7/20 + log(5)
            >>> from sympy import sstr
            >>> print(sstr((s.evalf(), e.evalf()), full_prec=True))
            (1.26629073187415, 0.0175000000000000)

        The endpoints may be symbolic:

            >>> s, e = Sum(1/k, (k, a, b)).euler_maclaurin()
            >>> s
            -log(a) + log(b) + 1/(2*b) + 1/(2*a)
            >>> e
            Abs(1/(12*b**2) - 1/(12*a**2))

        If the function is a polynomial of degree at most 2n+1, the
        Euler-Maclaurin formula becomes exact (and e = 0 is returned):

            >>> Sum(k, (k, 2, b)).euler_maclaurin()
            (b**2/2 + b/2 - 1, 0)
            >>> Sum(k, (k, 2, b)).doit()
            b**2/2 + b/2 - 1

        With a nonzero eps specified, the summation is ended
        as soon as the remainder term is less than the epsilon.
        """
    m = int(m)
    n = int(n)
    f = self.function
    if len(self.limits) != 1:
        raise ValueError('More than 1 limit')
    i, a, b = self.limits[0]
    if (a > b) == True:
        if a - b == 1:
            return (S.Zero, S.Zero)
        a, b = (b + 1, a - 1)
        f = -f
    s = S.Zero
    if m:
        if b.is_Integer and a.is_Integer:
            m = min(m, b - a + 1)
        if not eps or f.is_polynomial(i):
            s = Add(*[f.subs(i, a + k) for k in range(m)])
        else:
            term = f.subs(i, a)
            if term:
                test = abs(term.evalf(3)) < eps
                if test == True:
                    return (s, abs(term))
                elif not test == False:
                    return (term, S.Zero)
            s = term
            for k in range(1, m):
                term = f.subs(i, a + k)
                if abs(term.evalf(3)) < eps and term != 0:
                    return (s, abs(term))
                s += term
        if b - a + 1 == m:
            return (s, S.Zero)
        a += m
    x = Dummy('x')
    I = Integral(f.subs(i, x), (x, a, b))
    if eval_integral:
        I = I.doit()
    s += I

    def fpoint(expr):
        if b is S.Infinity:
            return (expr.subs(i, a), 0)
        return (expr.subs(i, a), expr.subs(i, b))
    fa, fb = fpoint(f)
    iterm = (fa + fb) / 2
    g = f.diff(i)
    for k in range(1, n + 2):
        ga, gb = fpoint(g)
        term = bernoulli(2 * k) / factorial(2 * k) * (gb - ga)
        if k > n:
            break
        if eps and term:
            term_evalf = term.evalf(3)
            if term_evalf is S.NaN:
                return (S.NaN, S.NaN)
            if abs(term_evalf) < eps:
                break
        s += term
        g = g.diff(i, 2, simplify=False)
    return (s + iterm, abs(term))