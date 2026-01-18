import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_single_vector():
    with pytest.warns(UserWarning, match='Optimal rotation is not'):
        r_estimate, rmsd = Rotation.align_vectors([[1, -1, 1]], [[1, 1, -1]])
        assert_allclose(rmsd, 0, atol=1e-16)