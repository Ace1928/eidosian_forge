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
def eval_sum_symbolic(f, limits):
    f_orig = f
    i, a, b = limits
    if not f.has(i):
        return f * (b - a + 1)
    if f.is_Mul:
        without_i, with_i = f.as_independent(i)
        if without_i != 1:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i * s
                if r is not S.NaN:
                    return r
        else:
            L, R = f.as_two_terms()
            if not L.has(i):
                sR = eval_sum_symbolic(R, (i, a, b))
                if sR:
                    return L * sR
            if not R.has(i):
                sL = eval_sum_symbolic(L, (i, a, b))
                if sL:
                    return sL * R
    try:
        f = apart(f, i)
    except PolynomialError:
        pass
    if f.is_Add:
        L, R = f.as_two_terms()
        lrsum = telescopic(L, R, (i, a, b))
        if lrsum:
            return lrsum
        without_i, with_i = f.as_independent(i)
        if without_i != 0:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i * (b - a + 1) + s
                if r is not S.NaN:
                    return r
        else:
            lsum = eval_sum_symbolic(L, (i, a, b))
            rsum = eval_sum_symbolic(R, (i, a, b))
            if None not in (lsum, rsum):
                r = lsum + rsum
                if r is not S.NaN:
                    return r
    n = Wild('n')
    result = f.match(i ** n)
    if result is not None:
        n = result[n]
        if n.is_Integer:
            if n >= 0:
                if b is S.Infinity and a is not S.NegativeInfinity or (a is S.NegativeInfinity and b is not S.Infinity):
                    return S.Infinity
                return ((bernoulli(n + 1, b + 1) - bernoulli(n + 1, a)) / (n + 1)).expand()
            elif a.is_Integer and a >= 1:
                if n == -1:
                    return harmonic(b) - harmonic(a - 1)
                else:
                    return harmonic(b, abs(n)) - harmonic(a - 1, abs(n))
    if not (a.has(S.Infinity, S.NegativeInfinity) or b.has(S.Infinity, S.NegativeInfinity)):
        c1 = Wild('c1', exclude=[i])
        c2 = Wild('c2', exclude=[i])
        c3 = Wild('c3', exclude=[i])
        wexp = Wild('wexp')
        e = f.powsimp().match(c1 ** wexp)
        if e is not None:
            e_exp = e.pop(wexp).expand().match(c2 * i + c3)
            if e_exp is not None:
                e.update(e_exp)
                p = (c1 ** c3).subs(e)
                q = (c1 ** c2).subs(e)
                r = p * (q ** a - q ** (b + 1)) / (1 - q)
                l = p * (b - a + 1)
                return Piecewise((l, Eq(q, S.One)), (r, True))
        r = gosper_sum(f, (i, a, b))
        if isinstance(r, (Mul, Add)):
            from sympy.simplify.radsimp import denom
            from sympy.solvers.solvers import solve
            non_limit = r.free_symbols - Tuple(*limits[1:]).free_symbols
            den = denom(together(r))
            den_sym = non_limit & den.free_symbols
            args = []
            for v in ordered(den_sym):
                try:
                    s = solve(den, v)
                    m = Eq(v, s[0]) if s else S.false
                    if m != False:
                        args.append((Sum(f_orig.subs(*m.args), limits).doit(), m))
                    break
                except NotImplementedError:
                    continue
            args.append((r, True))
            return Piecewise(*args)
        if r not in (None, S.NaN):
            return r
    h = eval_sum_hyper(f_orig, (i, a, b))
    if h is not None:
        return h
    r = eval_sum_residue(f_orig, (i, a, b))
    if r is not None:
        return r
    factored = f_orig.factor()
    if factored != f_orig:
        return eval_sum_symbolic(factored, (i, a, b))