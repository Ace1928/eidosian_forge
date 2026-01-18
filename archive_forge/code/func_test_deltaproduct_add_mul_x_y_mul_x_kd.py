from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises
def test_deltaproduct_add_mul_x_y_mul_x_kd():
    assert dp(x * y + x * KD(i, j), (j, 1, 3)) == (x * y) ** 3 + x * (x * y) ** 2 * KD(i, 1) + x * y * x * (x * y) * KD(i, 2) + (x * y) ** 2 * x * KD(i, 3)
    assert dp(x * y + x * KD(i, j), (j, 1, 1)) == x * y + x * KD(i, 1)
    assert dp(x * y + x * KD(i, j), (j, 2, 2)) == x * y + x * KD(i, 2)
    assert dp(x * y + x * KD(i, j), (j, 3, 3)) == x * y + x * KD(i, 3)
    assert dp(x * y + x * KD(i, j), (j, 1, k)) == (x * y) ** k + Piecewise(((x * y) ** (i - 1) * x * (x * y) ** (k - i), And(1 <= i, i <= k)), (0, True))
    assert dp(x * y + x * KD(i, j), (j, k, 3)) == (x * y) ** (-k + 4) + Piecewise(((x * y) ** (i - k) * x * (x * y) ** (3 - i), And(k <= i, i <= 3)), (0, True))
    assert dp(x * y + x * KD(i, j), (j, k, l)) == (x * y) ** (-k + l + 1) + Piecewise(((x * y) ** (i - k) * x * (x * y) ** (l - i), And(k <= i, i <= l)), (0, True))