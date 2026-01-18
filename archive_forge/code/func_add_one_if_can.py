from sympy.strategies.traverse import (
from sympy.strategies.rl import rebuild
from sympy.strategies.util import expr_fns
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Str, Symbol
from sympy.abc import x, y, z
def add_one_if_can(expr):
    try:
        return expr + 1
    except TypeError:
        return expr