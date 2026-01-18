import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_approx_equal_single_rotation():
    p = Rotation.from_rotvec([0, 0, 1e-09])
    q = Rotation.from_quat(np.eye(4))
    assert p.approx_equal(q[3])
    assert not p.approx_equal(q[0])
    assert not p.approx_equal(q[3], atol=1e-10)
    assert not p.approx_equal(q[3], atol=1e-08, degrees=True)
    with pytest.warns(UserWarning, match='atol must be set'):
        assert p.approx_equal(q[3], degrees=True)