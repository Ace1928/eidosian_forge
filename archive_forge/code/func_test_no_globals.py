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
def test_no_globals():
    default_globals = {}
    exec('from sympy import *', default_globals)
    builtins_dict = vars(builtins)
    for name, obj in builtins_dict.items():
        if isinstance(obj, types.BuiltinFunctionType):
            default_globals[name] = obj
    default_globals['max'] = Max
    default_globals['min'] = Min
    default_globals.pop('Symbol')
    global_dict = {'Symbol': Symbol}
    for name in default_globals:
        obj = parse_expr(name, global_dict=global_dict)
        assert obj == Symbol(name)