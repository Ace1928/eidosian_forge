from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises
def test_deltaproduct_mul_add_x_kd_add_y_kd():
    assert dp((x + KD(i, k)) * (y + KD(i, j)), (j, 1, 3)) == KD(i, 1) * (KD(i, k) + x) * ((KD(i, k) + x) * y) ** 2 + KD(i, 2) * (KD(i, k) + x) * y * (KD(i, k) + x) ** 2 * y + KD(i, 3) * ((KD(i, k) + x) * y) ** 2 * (KD(i, k) + x) + ((KD(i, k) + x) * y) ** 3
    assert dp((x + KD(i, k)) * (y + KD(i, j)), (j, 1, 1)) == (x + KD(i, k)) * (y + KD(i, 1))
    assert dp((x + KD(i, k)) * (y + KD(i, j)), (j, 2, 2)) == (x + KD(i, k)) * (y + KD(i, 2))
    assert dp((x + KD(i, k)) * (y + KD(i, j)), (j, 3, 3)) == (x + KD(i, k)) * (y + KD(i, 3))
    assert dp((x + KD(i, k)) * (y + KD(i, j)), (j, 1, k)) == ((KD(i, k) + x) * y) ** k + Piecewise((((KD(i, k) + x) * y) ** (-1) * ((KD(i, k) + x) * y) ** i * (KD(i, k) + x) * ((KD(i, k) + x) * y) ** k * ((KD(i, k) + x) * y) ** (-i), (i >= 1) & (i <= k)), (0, True))
    assert dp((x + KD(i, k)) * (y + KD(i, j)), (j, k, 3)) == ((KD(i, k) + x) * y) ** 4 * ((KD(i, k) + x) * y) ** (-k) + Piecewise((((KD(i, k) + x) * y) ** i * ((KD(i, k) + x) * y) ** (-k) * (KD(i, k) + x) * ((KD(i, k) + x) * y) ** 3 * ((KD(i, k) + x) * y) ** (-i), (i >= k) & (i <= 3)), (0, True))
    assert dp((x + KD(i, k)) * (y + KD(i, j)), (j, k, l)) == (KD(i, k) + x) * y * ((KD(i, k) + x) * y) ** l * ((KD(i, k) + x) * y) ** (-k) + Piecewise((((KD(i, k) + x) * y) ** i * ((KD(i, k) + x) * y) ** (-k) * (KD(i, k) + x) * ((KD(i, k) + x) * y) ** l * ((KD(i, k) + x) * y) ** (-i), (i >= k) & (i <= l)), (0, True))