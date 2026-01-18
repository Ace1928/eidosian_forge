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
def test_NormalQMCSeeded(self):
    seed = np.random.default_rng(274600237797326520096085022671371676017)
    engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), inv_transform=False, seed=seed)
    samples = engine.random(n=2)
    samples_expected = np.array([[-0.932001, -0.522923], [-1.477655, 0.846851]])
    assert_allclose(samples, samples_expected, atol=0.0001)
    seed = np.random.default_rng(274600237797326520096085022671371676017)
    engine = qmc.MultivariateNormalQMC(mean=np.zeros(3), inv_transform=False, seed=seed)
    samples = engine.random(n=2)
    samples_expected = np.array([[-0.932001, -0.522923, 0.036578], [-1.778011, 0.912428, -0.065421]])
    assert_allclose(samples, samples_expected, atol=0.0001)
    seed = np.random.default_rng(274600237797326520096085022671371676017)
    base_engine = qmc.Sobol(4, scramble=True, seed=seed)
    engine = qmc.MultivariateNormalQMC(mean=np.zeros(3), inv_transform=False, engine=base_engine, seed=seed)
    samples = engine.random(n=2)
    samples_expected = np.array([[-0.932001, -0.522923, 0.036578], [-1.778011, 0.912428, -0.065421]])
    assert_allclose(samples, samples_expected, atol=0.0001)