import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_euler_rotation_order():
    rnd = np.random.RandomState(0)
    a = rnd.randint(low=0, high=180, size=(6, 3))
    b = a[:, ::-1]
    x = Rotation.from_euler('xyz', a, degrees=True).as_quat()
    y = Rotation.from_euler('ZYX', b, degrees=True).as_quat()
    assert_allclose(x, y)