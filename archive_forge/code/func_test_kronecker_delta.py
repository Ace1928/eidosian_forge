from sympy.core.relational import Ne
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.tensor_functions import (Eijk, KroneckerDelta, LeviCivita)
from sympy.physics.secondquant import evaluate_deltas, F
def test_kronecker_delta():
    i, j = symbols('i j')
    k = Symbol('k', nonzero=True)
    assert KroneckerDelta(1, 1) == 1
    assert KroneckerDelta(1, 2) == 0
    assert KroneckerDelta(k, 0) == 0
    assert KroneckerDelta(x, x) == 1
    assert KroneckerDelta(x ** 2 - y ** 2, x ** 2 - y ** 2) == 1
    assert KroneckerDelta(i, i) == 1
    assert KroneckerDelta(i, i + 1) == 0
    assert KroneckerDelta(0, 0) == 1
    assert KroneckerDelta(0, 1) == 0
    assert KroneckerDelta(i + k, i) == 0
    assert KroneckerDelta(i + k, i + k) == 1
    assert KroneckerDelta(i + k, i + 1 + k) == 0
    assert KroneckerDelta(i, j).subs({'i': 1, 'j': 0}) == 0
    assert KroneckerDelta(i, j).subs({'i': 3, 'j': 3}) == 1
    assert KroneckerDelta(i, j) ** 0 == 1
    for n in range(1, 10):
        assert KroneckerDelta(i, j) ** n == KroneckerDelta(i, j)
        assert KroneckerDelta(i, j) ** (-n) == 1 / KroneckerDelta(i, j)
    assert KroneckerDelta(i, j).is_integer is True
    assert adjoint(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    assert conjugate(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    assert transpose(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    assert (KroneckerDelta(i, j) == KroneckerDelta(j, i)) == True
    assert KroneckerDelta(i, j).rewrite(Piecewise) == Piecewise((0, Ne(i, j)), (1, True))
    assert KroneckerDelta(i, j, (0, i)).args == (i, j, (0, i))
    assert KroneckerDelta(i, j, (-j, i)).delta_range == (-j, i)
    assert KroneckerDelta(i, j, (0, i - 1)) == 0
    assert KroneckerDelta(-1, j, (0, i - 1)) == 0
    assert KroneckerDelta(j, -1, (0, i - 1)) == 0
    assert KroneckerDelta(j, i, (0, i - 1)) == 0