import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_align_constrain():
    atol = 1e-12
    b = [[1, 0, 0], [1, 1, 0]]
    a = [[0, 1, 0], [0, 1, 1]]
    m_expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    R, rssd = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    assert_allclose(R.apply(b), a, atol=atol)
    assert np.isclose(rssd, 0, atol=atol)
    b = [[1, 0, 0], [1, 2, 0]]
    rssd_expected = 1.0
    R, rssd = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    assert_allclose(R.apply(b)[0], a[0], atol=atol)
    assert np.isclose(rssd, rssd_expected, atol=atol)
    a_expected = [[0, 1, 0], [0, 1, 2]]
    assert_allclose(R.apply(b), a_expected, atol=atol)
    b = [[1, 2, 3], [-2, 3, -1]]
    a = [[-1, 3, 2], [1, -1, 2]]
    rssd_expected = 1.3101595297515016
    R, rssd = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    assert_allclose(R.apply(b)[0], a[0], atol=atol)
    assert np.isclose(rssd, rssd_expected, atol=atol)