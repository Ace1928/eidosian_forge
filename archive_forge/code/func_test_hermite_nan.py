import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData
@pytest.mark.parametrize('n', [0, 1, 2])
@pytest.mark.parametrize('x', [0, 1, np.nan])
def test_hermite_nan(n, x):
    assert np.isnan(_ufuncs.eval_hermite(n, x)) == np.any(np.isnan([n, x]))
    assert np.isnan(_ufuncs.eval_hermitenorm(n, x)) == np.any(np.isnan([n, x]))