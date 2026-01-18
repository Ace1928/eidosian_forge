import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_NormalQMCSeededInvTransform(self):
    seed = np.random.default_rng(288527772707286126646493545351112463929)
    engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), seed=seed, inv_transform=True)
    samples = engine.random(n=2)
    samples_expected = np.array([[-0.913237, -0.964026], [0.255904, 0.003068]])
    assert_allclose(samples, samples_expected, atol=0.0001)
    seed = np.random.default_rng(288527772707286126646493545351112463929)
    engine = qmc.MultivariateNormalQMC(mean=np.zeros(3), seed=seed, inv_transform=True)
    samples = engine.random(n=2)
    samples_expected = np.array([[-0.913237, -0.964026, 0.355501], [0.699261, 2.90213, -0.6418]])
    assert_allclose(samples, samples_expected, atol=0.0001)