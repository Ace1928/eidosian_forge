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
def test_local_dict():
    local_dict = {'my_function': lambda x: x + 2}
    inputs = {'my_function(2)': Integer(4)}
    for text, result in inputs.items():
        assert parse_expr(text, local_dict=local_dict) == result