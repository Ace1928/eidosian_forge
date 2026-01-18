from sympy.assumptions.ask import Q
from sympy.assumptions.wrapper import (AssumptionsWrapper, is_infinite,
from sympy.core.symbol import Symbol
from sympy.core.assumptions import _assume_defined
def test_is_infinite():
    x = Symbol('x', infinite=True)
    y = Symbol('y', infinite=False)
    z = Symbol('z')
    assert is_infinite(x)
    assert not is_infinite(y)
    assert is_infinite(z) is None
    assert is_infinite(z, Q.infinite(z))