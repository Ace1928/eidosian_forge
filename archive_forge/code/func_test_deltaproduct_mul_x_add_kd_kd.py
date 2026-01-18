from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises
def test_deltaproduct_mul_x_add_kd_kd():
    assert dp(x * (KD(i, k) + KD(j, k)), (k, 1, 3)) == 0
    assert dp(x * (KD(i, k) + KD(j, k)), (k, 1, 1)) == x * (KD(i, 1) + KD(j, 1))
    assert dp(x * (KD(i, k) + KD(j, k)), (k, 2, 2)) == x * (KD(i, 2) + KD(j, 2))
    assert dp(x * (KD(i, k) + KD(j, k)), (k, 3, 3)) == x * (KD(i, 3) + KD(j, 3))
    assert dp(x * (KD(i, k) + KD(j, k)), (k, 1, l)) == KD(l, 0) + x * KD(i, 1) * KD(l, 1) + x * KD(j, 1) * KD(l, 1) + x ** 2 * KD(i, 1) * KD(j, 2) * KD(l, 2) + x ** 2 * KD(j, 1) * KD(i, 2) * KD(l, 2)
    assert dp(x * (KD(i, k) + KD(j, k)), (k, l, 3)) == KD(l, 4) + x * KD(i, 3) * KD(l, 3) + x * KD(j, 3) * KD(l, 3) + x ** 2 * KD(i, 2) * KD(j, 3) * KD(l, 2) + x ** 2 * KD(i, 3) * KD(j, 2) * KD(l, 2)
    assert dp(x * (KD(i, k) + KD(j, k)), (k, l, m)) == KD(l, m + 1) + x * KD(i, m) * KD(l, m) + x * KD(j, m) * KD(l, m) + x ** 2 * KD(i, m - 1) * KD(j, m) * KD(l, m - 1) + x ** 2 * KD(i, m) * KD(j, m - 1) * KD(l, m - 1)