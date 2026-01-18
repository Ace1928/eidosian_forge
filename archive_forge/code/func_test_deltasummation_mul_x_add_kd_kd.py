from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises
def test_deltasummation_mul_x_add_kd_kd():
    assert ds(x * (KD(i, k) + KD(j, k)), (k, 1, 3)) == piecewise_fold(Piecewise((x, And(1 <= i, i <= 3)), (0, True)) + Piecewise((x, And(1 <= j, j <= 3)), (0, True)))
    assert ds(x * (KD(i, k) + KD(j, k)), (k, 1, 1)) == piecewise_fold(Piecewise((x, Eq(i, 1)), (0, True)) + Piecewise((x, Eq(j, 1)), (0, True)))
    assert ds(x * (KD(i, k) + KD(j, k)), (k, 2, 2)) == piecewise_fold(Piecewise((x, Eq(i, 2)), (0, True)) + Piecewise((x, Eq(j, 2)), (0, True)))
    assert ds(x * (KD(i, k) + KD(j, k)), (k, 3, 3)) == piecewise_fold(Piecewise((x, Eq(i, 3)), (0, True)) + Piecewise((x, Eq(j, 3)), (0, True)))
    assert ds(x * (KD(i, k) + KD(j, k)), (k, 1, l)) == piecewise_fold(Piecewise((x, And(1 <= i, i <= l)), (0, True)) + Piecewise((x, And(1 <= j, j <= l)), (0, True)))
    assert ds(x * (KD(i, k) + KD(j, k)), (k, l, 3)) == piecewise_fold(Piecewise((x, And(l <= i, i <= 3)), (0, True)) + Piecewise((x, And(l <= j, j <= 3)), (0, True)))
    assert ds(x * (KD(i, k) + KD(j, k)), (k, l, m)) == piecewise_fold(Piecewise((x, And(l <= i, i <= m)), (0, True)) + Piecewise((x, And(l <= j, j <= m)), (0, True)))