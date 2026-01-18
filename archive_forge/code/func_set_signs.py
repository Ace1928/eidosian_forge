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
def set_signs(expr):
    if not expr.args:
        return expr
    newargs = tuple((set_signs(arg) for arg in expr.args))
    if newargs != expr.args:
        expr = expr.func(*newargs)
    abs_flag = isinstance(expr, Abs)
    arg_flag = isinstance(expr, arg)
    sign_flag = isinstance(expr, sign)
    if abs_flag or sign_flag or arg_flag:
        sig = limit(expr.args[0], z, z0, dir)
        if sig.is_zero:
            sig = limit(1 / expr.args[0], z, z0, dir)
        if sig.is_extended_real:
            if (sig < 0) == True:
                return -expr.args[0] if abs_flag else S.NegativeOne if sign_flag else S.Pi
            elif (sig > 0) == True:
                return expr.args[0] if abs_flag else S.One if sign_flag else S.Zero
    return expr