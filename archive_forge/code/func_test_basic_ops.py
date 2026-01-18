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
def test_basic_ops():
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(x * y, auto_declare=False, log_warn=w) == '(* x y)'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(x + y, auto_declare=False, log_warn=w) == '(+ x y)'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(-x, auto_declare=False, log_warn=w) == '(* -1 x)'