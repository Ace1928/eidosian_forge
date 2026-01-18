import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_mean_invalid_weights():
    with pytest.raises(ValueError, match='non-negative'):
        r = Rotation.from_quat(np.eye(4))
        r.mean(weights=-np.ones(4))