from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def exptrigsimp(expr):
    """
    Simplifies exponential / trigonometric / hyperbolic functions.

    Examples
    ========

    >>> from sympy import exptrigsimp, exp, cosh, sinh
    >>> from sympy.abc import z

    >>> exptrigsimp(exp(z) + exp(-z))
    2*cosh(z)
    >>> exptrigsimp(cosh(z) - sinh(z))
    exp(-z)
    """
    from sympy.simplify.fu import hyper_as_trig, TR2i

    def exp_trig(e):
        choices = [e]
        if e.has(*_trigs):
            choices.append(e.rewrite(exp))
        choices.append(e.rewrite(cos))
        return min(*choices, key=count_ops)
    newexpr = bottom_up(expr, exp_trig)

    def f(rv):
        if not rv.is_Mul:
            return rv
        commutative_part, noncommutative_part = rv.args_cnc()
        if len(noncommutative_part) > 1:
            return f(Mul(*commutative_part)) * Mul(*noncommutative_part)
        rvd = rv.as_powers_dict()
        newd = rvd.copy()

        def signlog(expr, sign=S.One):
            if expr is S.Exp1:
                return (sign, S.One)
            elif isinstance(expr, exp) or (expr.is_Pow and expr.base == S.Exp1):
                return (sign, expr.exp)
            elif sign is S.One:
                return signlog(-expr, sign=-S.One)
            else:
                return (None, None)
        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                c = k.args[0]
                sign, x = signlog(k.args[1] / c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x * m / 2:
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2 * c * cosh(x / 2)] += m
                    else:
                        newd[-2 * c * sinh(x / 2)] += m
                elif newd[1 - sign * S.Exp1 ** x] == -m:
                    del newd[1 - sign * S.Exp1 ** x]
                    if sign == 1:
                        newd[-c / tanh(x / 2)] += m
                    else:
                        newd[-c * tanh(x / 2)] += m
                else:
                    newd[1 + sign * S.Exp1 ** x] += m
                    newd[c] += m
        return Mul(*[k ** newd[k] for k in newd])
    newexpr = bottom_up(newexpr, f)
    if newexpr.has(HyperbolicFunction):
        e, f = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)
    if not (newexpr.has(I) and (not expr.has(I))):
        expr = newexpr
    return expr