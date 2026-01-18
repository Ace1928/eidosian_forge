import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
@pytest.mark.parametrize('dtype', ('f4', 'f8'))
def test_fillpositive_plus_minus_epsilon(dtype):
    nptype = np.dtype(dtype).type
    baseline = np.array([0, 0, 1], dtype=dtype)
    plus = baseline * nptype(1 + np.finfo(dtype).eps)
    minus = baseline * nptype(1 - np.finfo(dtype).eps)
    assert nq.fillpositive(plus)[0] == 0.0
    assert nq.fillpositive(minus)[0] == 0.0
    plus = baseline * nptype(1 + 2 * np.finfo(dtype).eps)
    with pytest.raises(ValueError):
        nq.fillpositive(plus)
    minus = baseline * nptype(1 - 2 * np.finfo(dtype).eps)
    assert nq.fillpositive(minus)[0] > 0.0