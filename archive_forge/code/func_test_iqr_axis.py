import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.tools.eval_measures import (
def test_iqr_axis(reset_randomstate):
    x1 = np.random.standard_normal((100, 100))
    x2 = np.random.standard_normal((100, 100))
    ax_none = iqr(x1, x2, axis=None)
    ax_none_direct = iqr(x1.ravel(), x2.ravel())
    assert_equal(ax_none, ax_none_direct)
    ax_0 = iqr(x1, x2, axis=0)
    assert ax_0.shape == (100,)
    ax_0_direct = [iqr(x1[:, i], x2[:, i]) for i in range(100)]
    assert_almost_equal(ax_0, np.array(ax_0_direct))
    ax_1 = iqr(x1, x2, axis=1)
    assert ax_1.shape == (100,)
    ax_1_direct = [iqr(x1[i, :], x2[i, :]) for i in range(100)]
    assert_almost_equal(ax_1, np.array(ax_1_direct))
    assert any(ax_0 != ax_1)