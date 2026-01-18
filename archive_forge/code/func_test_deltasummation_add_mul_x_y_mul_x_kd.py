from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises
def test_deltasummation_add_mul_x_y_mul_x_kd():
    assert ds(x * y + x * KD(i, j), (j, 1, 3)) == Piecewise((3 * x * y + x, And(1 <= i, i <= 3)), (3 * x * y, True))
    assert ds(x * y + x * KD(i, j), (j, 1, 1)) == Piecewise((x * y + x, Eq(i, 1)), (x * y, True))
    assert ds(x * y + x * KD(i, j), (j, 2, 2)) == Piecewise((x * y + x, Eq(i, 2)), (x * y, True))
    assert ds(x * y + x * KD(i, j), (j, 3, 3)) == Piecewise((x * y + x, Eq(i, 3)), (x * y, True))
    assert ds(x * y + x * KD(i, j), (j, 1, k)) == Piecewise((k * x * y + x, And(1 <= i, i <= k)), (k * x * y, True))
    assert ds(x * y + x * KD(i, j), (j, k, 3)) == Piecewise(((4 - k) * x * y + x, And(k <= i, i <= 3)), ((4 - k) * x * y, True))
    assert ds(x * y + x * KD(i, j), (j, k, l)) == Piecewise(((l - k + 1) * x * y + x, And(k <= i, i <= l)), ((l - k + 1) * x * y, True))