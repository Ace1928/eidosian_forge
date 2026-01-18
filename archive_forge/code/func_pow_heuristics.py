from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import S, Symbol, Add, sympify, Expr, PoleError, Mul
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Float, _illegal
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, sign, arg, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.special.gamma_functions import gamma
from sympy.polys import PolynomialError, factor
from sympy.series.order import Order
from .gruntz import gruntz
def pow_heuristics(self, e):
    _, z, z0, _ = self.args
    b1, e1 = (e.base, e.exp)
    if not b1.has(z):
        res = limit(e1 * log(b1), z, z0)
        return exp(res)
    ex_lim = limit(e1, z, z0)
    base_lim = limit(b1, z, z0)
    if base_lim is S.One:
        if ex_lim in (S.Infinity, S.NegativeInfinity):
            res = limit(e1 * (b1 - 1), z, z0)
            return exp(res)
    if base_lim is S.NegativeInfinity and ex_lim is S.Infinity:
        return S.ComplexInfinity