import contextlib
import itertools
import re
import typing
from enum import Enum
from typing import Callable
import sympy
from sympy import Add, Implies, sqrt
from sympy.core import Mul, Pow
from sympy.core import (S, pi, symbols, Function, Rational, Integer,
from sympy.functions import Piecewise, exp, sin, cos
from sympy.printing.smtlib import smtlib_code
from sympy.testing.pytest import raises, Failed
def test_Relational():
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 12) as w:
        assert smtlib_code(Eq(x, y), auto_declare=False, log_warn=w) == '(assert (= x y))'
        assert smtlib_code(Ne(x, y), auto_declare=False, log_warn=w) == '(assert (not (= x y)))'
        assert smtlib_code(Le(x, y), auto_declare=False, log_warn=w) == '(assert (<= x y))'
        assert smtlib_code(Lt(x, y), auto_declare=False, log_warn=w) == '(assert (< x y))'
        assert smtlib_code(Gt(x, y), auto_declare=False, log_warn=w) == '(assert (> x y))'
        assert smtlib_code(Ge(x, y), auto_declare=False, log_warn=w) == '(assert (>= x y))'