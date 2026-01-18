from sympy.core.singleton import S
from sympy.strategies.rl import (
from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.abc import x
def test_glom():

    def key(x):
        return x.as_coeff_Mul()[1]

    def count(x):
        return x.as_coeff_Mul()[0]

    def newargs(cnt, arg):
        return cnt * arg
    rl = glom(key, count, newargs)
    result = rl(Add(x, -x, 3 * x, 2, 3, evaluate=False))
    expected = Add(3 * x, 5)
    assert set(result.args) == set(expected.args)