from sympy.physics.secondquant import (
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.repr import srepr
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import slow, raises
from sympy.printing.latex import latex
def test_fixed_bosonic_basis():
    b = FixedBosonicBasis(2, 2)
    state = b.state(1)
    assert state == FockStateBosonKet((1, 1))
    assert b.index(state) == 1
    assert b.state(1) == b[1]
    assert len(b) == 3
    assert str(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'
    assert repr(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'
    assert srepr(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'