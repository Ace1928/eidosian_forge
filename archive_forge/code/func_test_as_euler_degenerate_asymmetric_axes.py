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
def test_as_euler_degenerate_asymmetric_axes(seq_tuple, intrinsic):
    angles = np.array([[45, 90, 35], [35, -90, 20], [35, 90, 25], [25, -90, 15]])
    seq = ''.join(seq_tuple)
    if intrinsic:
        seq = seq.upper()
    rotation = Rotation.from_euler(seq, angles, degrees=True)
    mat_expected = rotation.as_matrix()
    with pytest.warns(UserWarning, match='Gimbal lock'):
        angle_estimates = rotation.as_euler(seq, degrees=True)
    mat_estimated = Rotation.from_euler(seq, angle_estimates, degrees=True).as_matrix()
    assert_array_almost_equal(mat_expected, mat_estimated)