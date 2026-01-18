import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
@pytest.mark.parametrize('dtype', ('f4', 'f8'))
def test_fillpositive_simulated_error(dtype):
    w2_thresh = 3 * np.finfo(dtype).eps
    pos_error = neg_error = False
    for _ in range(50):
        xyz = norm(gen_vec(dtype))
        assert nq.fillpositive(xyz, w2_thresh)[0] == 0.0