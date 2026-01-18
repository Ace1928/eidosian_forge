import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_noise():
    rnd = np.random.RandomState(0)
    n_vectors = 100
    rot = Rotation.random(random_state=rnd)
    vectors = rnd.normal(size=(n_vectors, 3))
    result = rot.apply(vectors)
    sigma = np.deg2rad(1)
    tolerance = 1.5 * sigma
    noise = Rotation.from_rotvec(rnd.normal(size=(n_vectors, 3), scale=sigma))
    noisy_result = noise.apply(result)
    est, rssd, cov = Rotation.align_vectors(noisy_result, vectors, return_sensitivity=True)
    error_vector = (rot * est.inv()).as_rotvec()
    assert_allclose(error_vector[0], 0, atol=tolerance)
    assert_allclose(error_vector[1], 0, atol=tolerance)
    assert_allclose(error_vector[2], 0, atol=tolerance)
    cov *= sigma
    assert_allclose(cov[0, 0], 0, atol=tolerance)
    assert_allclose(cov[1, 1], 0, atol=tolerance)
    assert_allclose(cov[2, 2], 0, atol=tolerance)
    assert_allclose(rssd, np.sum((noisy_result - est.apply(vectors)) ** 2) ** 0.5)