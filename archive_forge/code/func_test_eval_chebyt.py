import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData
def test_eval_chebyt():
    n = np.arange(0, 10000, 7, dtype=np.dtype('long'))
    x = 2 * np.random.rand() - 1
    v1 = np.cos(n * np.arccos(x))
    v2 = _ufuncs.eval_chebyt(n, x)
    assert_(np.allclose(v1, v2, rtol=1e-15))