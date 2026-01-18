import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
@pytest.mark.parametrize('seq_tuple', permutations('xyz'))
@pytest.mark.parametrize('intrinsic', (False, True))
def test_as_euler_asymmetric_axes(seq_tuple, intrinsic):

    def test_stats(error, mean_max, rms_max):
        mean = np.mean(error, axis=0)
        std = np.std(error, axis=0)
        rms = np.hypot(mean, std)
        assert np.all(np.abs(mean) < mean_max)
        assert np.all(rms < rms_max)
    rnd = np.random.RandomState(0)
    n = 1000
    angles = np.empty((n, 3))
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles[:, 1] = rnd.uniform(low=-np.pi / 2, high=np.pi / 2, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    seq = ''.join(seq_tuple)
    if intrinsic:
        seq = seq.upper()
    rotation = Rotation.from_euler(seq, angles)
    angles_quat = rotation.as_euler(seq)
    angles_mat = rotation._as_euler_from_matrix(seq)
    assert_allclose(angles, angles_quat, atol=0, rtol=1e-12)
    assert_allclose(angles, angles_mat, atol=0, rtol=1e-12)
    test_stats(angles_quat - angles, 1e-15, 1e-14)
    test_stats(angles_mat - angles, 1e-15, 1e-14)