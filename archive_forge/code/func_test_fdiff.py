from sympy.core.function import (Derivative, diff)
from sympy.core.numbers import (Float, I, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
def test_fdiff():
    assert SingularityFunction(x, 4, 5).fdiff() == 5 * SingularityFunction(x, 4, 4)
    assert SingularityFunction(x, 4, -1).fdiff() == SingularityFunction(x, 4, -2)
    assert SingularityFunction(x, 4, 0).fdiff() == SingularityFunction(x, 4, -1)
    assert SingularityFunction(y, 6, 2).diff(y) == 2 * SingularityFunction(y, 6, 1)
    assert SingularityFunction(y, -4, -1).diff(y) == SingularityFunction(y, -4, -2)
    assert SingularityFunction(y, 4, 0).diff(y) == SingularityFunction(y, 4, -1)
    assert SingularityFunction(y, 4, 0).diff(y, 2) == SingularityFunction(y, 4, -2)
    n = Symbol('n', positive=True)
    assert SingularityFunction(x, a, n).fdiff() == n * SingularityFunction(x, a, n - 1)
    assert SingularityFunction(y, a, n).diff(y) == n * SingularityFunction(y, a, n - 1)
    expr_in = 4 * SingularityFunction(x, a, n) + 3 * SingularityFunction(x, a, -1) + -10 * SingularityFunction(x, a, 0)
    expr_out = n * 4 * SingularityFunction(x, a, n - 1) + 3 * SingularityFunction(x, a, -2) - 10 * SingularityFunction(x, a, -1)
    assert diff(expr_in, x) == expr_out
    assert SingularityFunction(x, -10, 5).diff(evaluate=False) == Derivative(SingularityFunction(x, -10, 5), x)
    raises(ArgumentIndexError, lambda: SingularityFunction(x, 4, 5).fdiff(2))