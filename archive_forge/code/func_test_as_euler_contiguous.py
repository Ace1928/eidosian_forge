import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_as_euler_contiguous():
    r = Rotation.from_quat([0, 0, 0, 1])
    e1 = r.as_euler('xyz')
    e2 = r.as_euler('XYZ')
    assert e1.flags['C_CONTIGUOUS'] is True
    assert e2.flags['C_CONTIGUOUS'] is True
    assert all((i >= 0 for i in e1.strides))
    assert all((i >= 0 for i in e2.strides))