import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def test_fillpos():
    xyz = np.zeros((3,))
    w, x, y, z = nq.fillpositive(xyz)
    assert w == 1
    xyz = [0] * 3
    w, x, y, z = nq.fillpositive(xyz)
    assert w == 1
    with pytest.raises(ValueError):
        nq.fillpositive([0, 0])
    with pytest.raises(ValueError):
        nq.fillpositive([0] * 4)
    with pytest.raises(ValueError):
        nq.fillpositive([1.0] * 3)
    wxyz = nq.fillpositive([1, 0, 0])
    assert wxyz[0] == 0.0