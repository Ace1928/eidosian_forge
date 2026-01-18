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
def test_Function():
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(sin(x) ** cos(x), auto_declare=False, log_warn=w) == '(pow (sin x) (cos x))'
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(abs(x), symbol_table={x: int, y: bool}, known_types={int: 'INTEGER_TYPE'}, known_functions={sympy.Abs: 'ABSOLUTE_VALUE_OF'}, log_warn=w) == '(declare-const x INTEGER_TYPE)\n(ABSOLUTE_VALUE_OF x)'
    my_fun1 = Function('f1')
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(my_fun1(x), symbol_table={my_fun1: Callable[[bool], float]}, log_warn=w) == '(declare-const x Bool)\n(declare-fun f1 (Bool) Real)\n(f1 x)'
    with _check_warns([]) as w:
        assert smtlib_code(my_fun1(x), symbol_table={my_fun1: Callable[[bool], bool]}, log_warn=w) == '(declare-const x Bool)\n(declare-fun f1 (Bool) Bool)\n(assert (f1 x))'
        assert smtlib_code(Eq(my_fun1(x, z), y), symbol_table={my_fun1: Callable[[int, bool], bool]}, log_warn=w) == '(declare-const x Int)\n(declare-const y Bool)\n(declare-const z Bool)\n(declare-fun f1 (Int Bool) Bool)\n(assert (= (f1 x z) y))'
        assert smtlib_code(Eq(my_fun1(x, z), y), symbol_table={my_fun1: Callable[[int, bool], bool]}, known_functions={my_fun1: 'MY_KNOWN_FUN', Eq: '=='}, log_warn=w) == '(declare-const x Int)\n(declare-const y Bool)\n(declare-const z Bool)\n(assert (== (MY_KNOWN_FUN x z) y))'
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 3) as w:
        assert smtlib_code(Eq(my_fun1(x, z), y), known_functions={my_fun1: 'MY_KNOWN_FUN', Eq: '=='}, log_warn=w) == '(declare-const x Real)\n(declare-const y Real)\n(declare-const z Real)\n(assert (== (MY_KNOWN_FUN x z) y))'