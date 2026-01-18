import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData
@pytest.mark.parametrize('n', [0, 1, 2, 3.2])
@pytest.mark.parametrize('alpha', [1, np.nan])
@pytest.mark.parametrize('x', [2, np.nan])
def test_genlaguerre_nan(n, alpha, x):
    nan_laguerre = np.isnan(_ufuncs.eval_genlaguerre(n, alpha, x))
    nan_arg = np.any(np.isnan([n, alpha, x]))
    assert nan_laguerre == nan_arg