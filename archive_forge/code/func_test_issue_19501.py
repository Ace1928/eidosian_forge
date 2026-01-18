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
def test_issue_19501():
    x = Symbol('x')
    eq = parse_expr('E**x(1+x)', local_dict={'x': x}, transformations=standard_transformations + (implicit_multiplication_application,))
    assert eq.free_symbols == {x}