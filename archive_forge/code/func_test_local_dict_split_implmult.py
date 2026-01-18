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
def test_local_dict_split_implmult():
    t = standard_transformations + (split_symbols, implicit_multiplication)
    w = Symbol('w', real=True)
    y = Symbol('y')
    assert parse_expr('yx', local_dict={'x': w}, transformations=t) == y * w