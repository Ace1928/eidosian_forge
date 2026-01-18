import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_davenport_invalid_input():
    ez = [0, 0, 1]
    ey = [0, 1, 0]
    ezy = [0, 1, 1]
    with pytest.raises(ValueError, match='must be orthogonal'):
        Rotation.from_davenport([ez, ezy], 'e', [0, 0])
    with pytest.raises(ValueError, match='must be orthogonal'):
        Rotation.from_davenport([ez, ey, ezy], 'e', [0, 0, 0])
    with pytest.raises(ValueError, match='order should be'):
        Rotation.from_davenport([ez], 'xyz', [0])
    with pytest.raises(ValueError, match='Expected `angles`'):
        Rotation.from_davenport([ez, ey, ez], 'e', [0, 1, 2, 3])