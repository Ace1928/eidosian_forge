from sympy.core.numbers import (I, nan, oo, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, sign, transpose)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
def test_DiracDelta():
    assert DiracDelta(1) == 0
    assert DiracDelta(5.1) == 0
    assert DiracDelta(-pi) == 0
    assert DiracDelta(5, 7) == 0
    assert DiracDelta(x, 0) == DiracDelta(x)
    assert DiracDelta(i) == 0
    assert DiracDelta(j) == 0
    assert DiracDelta(k) == 0
    assert DiracDelta(nan) is nan
    assert DiracDelta(0).func is DiracDelta
    assert DiracDelta(x).func is DiracDelta
    assert adjoint(DiracDelta(x)) == DiracDelta(x)
    assert adjoint(DiracDelta(x - y)) == DiracDelta(x - y)
    assert conjugate(DiracDelta(x)) == DiracDelta(x)
    assert conjugate(DiracDelta(x - y)) == DiracDelta(x - y)
    assert transpose(DiracDelta(x)) == DiracDelta(x)
    assert transpose(DiracDelta(x - y)) == DiracDelta(x - y)
    assert DiracDelta(x).diff(x) == DiracDelta(x, 1)
    assert DiracDelta(x, 1).diff(x) == DiracDelta(x, 2)
    assert DiracDelta(x).is_simple(x) is True
    assert DiracDelta(3 * x).is_simple(x) is True
    assert DiracDelta(x ** 2).is_simple(x) is False
    assert DiracDelta(sqrt(x)).is_simple(x) is False
    assert DiracDelta(x).is_simple(y) is False
    assert DiracDelta(x * y).expand(diracdelta=True, wrt=x) == DiracDelta(x) / abs(y)
    assert DiracDelta(x * y).expand(diracdelta=True, wrt=y) == DiracDelta(y) / abs(x)
    assert DiracDelta(x ** 2 * y).expand(diracdelta=True, wrt=x) == DiracDelta(x ** 2 * y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1) * (x - 2) * (x - 3)).expand(diracdelta=True, wrt=x) == DiracDelta(x - 3) / 2 + DiracDelta(x - 2) + DiracDelta(x - 1) / 2
    assert DiracDelta(2 * x) != DiracDelta(x)
    assert DiracDelta(x) == DiracDelta(-x)
    assert DiracDelta(-x, 2) == DiracDelta(x, 2)
    assert DiracDelta(-x, 1) == -DiracDelta(x, 1)
    assert DiracDelta(-oo * x) == DiracDelta(oo * x)
    assert DiracDelta(x - y) != DiracDelta(y - x)
    assert signsimp(DiracDelta(x - y) - DiracDelta(y - x)) == 0
    assert DiracDelta(x * y).expand(diracdelta=True, wrt=x) == DiracDelta(x) / abs(y)
    assert DiracDelta(x * y).expand(diracdelta=True, wrt=y) == DiracDelta(y) / abs(x)
    assert DiracDelta(x ** 2 * y).expand(diracdelta=True, wrt=x) == DiracDelta(x ** 2 * y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1) * (x - 2) * (x - 3)).expand(diracdelta=True) == DiracDelta(x - 3) / 2 + DiracDelta(x - 2) + DiracDelta(x - 1) / 2
    raises(ArgumentIndexError, lambda: DiracDelta(x).fdiff(2))
    raises(ValueError, lambda: DiracDelta(x, -1))
    raises(ValueError, lambda: DiracDelta(I))
    raises(ValueError, lambda: DiracDelta(2 + 3 * I))