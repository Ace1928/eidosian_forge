from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import (Symbol, symbols)
from sympy.testing.pytest import XFAIL
def test_constructor_postprocessors1():
    x = SymbolInMulOnce('x')
    y = SymbolInMulOnce('y')
    assert isinstance(3 * x, Mul)
    assert (3 * x).args == (3, x)
    assert x * x == x
    assert 3 * x * x == 3 * x
    assert 2 * x * x + x == 3 * x
    assert x ** 3 * y * y == x * y
    assert x ** 5 + y * x ** 3 == x + x * y
    w = SymbolRemovesOtherSymbols('w')
    assert x * w == w
    assert (3 * w).args == (3, w)
    assert set((w + x).args) == {x, w}