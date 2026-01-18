from sympy.assumptions.ask import Q
from sympy.assumptions.wrapper import (AssumptionsWrapper, is_infinite,
from sympy.core.symbol import Symbol
from sympy.core.assumptions import _assume_defined
def test_is_extended_real():
    x = Symbol('x', extended_real=True)
    y = Symbol('y', extended_real=False)
    z = Symbol('z')
    assert is_extended_real(x)
    assert not is_extended_real(y)
    assert is_extended_real(z) is None
    assert is_extended_real(z, Q.extended_real(z))