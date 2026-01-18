import sys
import builtins
import types
from sympy.assumptions import Q
from sympy.core import Symbol, Function, Float, Rational, Integer, I, Mul, Pow, Eq, Lt, Le, Gt, Ge, Ne
from sympy.functions import exp, factorial, factorial2, sin, Min, Max
from sympy.logic import And
from sympy.series import Limit
from sympy.testing.pytest import raises, skip
from sympy.parsing.sympy_parser import (
def test_function_evaluate_false():
    inputs = ['Abs(0)', 'im(0)', 're(0)', 'sign(0)', 'arg(0)', 'conjugate(0)', 'acos(0)', 'acot(0)', 'acsc(0)', 'asec(0)', 'asin(0)', 'atan(0)', 'acosh(0)', 'acoth(0)', 'acsch(0)', 'asech(0)', 'asinh(0)', 'atanh(0)', 'cos(0)', 'cot(0)', 'csc(0)', 'sec(0)', 'sin(0)', 'tan(0)', 'cosh(0)', 'coth(0)', 'csch(0)', 'sech(0)', 'sinh(0)', 'tanh(0)', 'exp(0)', 'log(0)', 'sqrt(0)']
    for case in inputs:
        expr = parse_expr(case, evaluate=False)
        assert case == str(expr) != str(expr.doit())
    assert str(parse_expr('ln(0)', evaluate=False)) == 'log(0)'
    assert str(parse_expr('cbrt(0)', evaluate=False)) == '0**(1/3)'