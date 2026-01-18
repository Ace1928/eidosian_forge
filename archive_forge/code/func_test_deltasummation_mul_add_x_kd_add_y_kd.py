from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises
def test_deltasummation_mul_add_x_kd_add_y_kd():
    assert ds((x + KD(i, k)) * (y + KD(i, j)), (j, 1, 3)) == piecewise_fold(Piecewise((KD(i, k) + x, And(1 <= i, i <= 3)), (0, True)) + 3 * (KD(i, k) + x) * y)
    assert ds((x + KD(i, k)) * (y + KD(i, j)), (j, 1, 1)) == piecewise_fold(Piecewise((KD(i, k) + x, Eq(i, 1)), (0, True)) + (KD(i, k) + x) * y)
    assert ds((x + KD(i, k)) * (y + KD(i, j)), (j, 2, 2)) == piecewise_fold(Piecewise((KD(i, k) + x, Eq(i, 2)), (0, True)) + (KD(i, k) + x) * y)
    assert ds((x + KD(i, k)) * (y + KD(i, j)), (j, 3, 3)) == piecewise_fold(Piecewise((KD(i, k) + x, Eq(i, 3)), (0, True)) + (KD(i, k) + x) * y)
    assert ds((x + KD(i, k)) * (y + KD(i, j)), (j, 1, k)) == piecewise_fold(Piecewise((KD(i, k) + x, And(1 <= i, i <= k)), (0, True)) + k * (KD(i, k) + x) * y)
    assert ds((x + KD(i, k)) * (y + KD(i, j)), (j, k, 3)) == piecewise_fold(Piecewise((KD(i, k) + x, And(k <= i, i <= 3)), (0, True)) + (4 - k) * (KD(i, k) + x) * y)
    assert ds((x + KD(i, k)) * (y + KD(i, j)), (j, k, l)) == piecewise_fold(Piecewise((KD(i, k) + x, And(k <= i, i <= l)), (0, True)) + (l - k + 1) * (KD(i, k) + x) * y)