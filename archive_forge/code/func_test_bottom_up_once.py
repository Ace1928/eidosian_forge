from sympy.strategies.traverse import (
from sympy.strategies.rl import rebuild
from sympy.strategies.util import expr_fns
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Str, Symbol
from sympy.abc import x, y, z
def test_bottom_up_once():
    bottom_rl = bottom_up_once(rl)
    assert bottom_rl(Basic(S(1), S(2), Basic(S(3.0), S(4.0)))) == Basic(S(1), S(2), Basic2(S(3.0), S(4.0)))