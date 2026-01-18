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
def test_MultivariateNormalQMCInvTransform(self):
    engine = qmc.MultivariateNormalQMC(mean=0, cov=5, inv_transform=True)
    samples = engine.random()
    assert_equal(samples.shape, (1, 1))
    samples = engine.random(n=5)
    assert_equal(samples.shape, (5, 1))
    engine = qmc.MultivariateNormalQMC(mean=[0, 1], cov=[[1, 0], [0, 1]], inv_transform=True)
    samples = engine.random()
    assert_equal(samples.shape, (1, 2))
    samples = engine.random(n=5)
    assert_equal(samples.shape, (5, 2))
    mean = np.array([0, 1, 2])
    cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    engine = qmc.MultivariateNormalQMC(mean, cov, inv_transform=True)
    samples = engine.random()
    assert_equal(samples.shape, (1, 3))
    samples = engine.random(n=5)
    assert_equal(samples.shape, (5, 3))