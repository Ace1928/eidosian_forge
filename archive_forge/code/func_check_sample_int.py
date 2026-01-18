import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb
from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement
def check_sample_int(sample_without_replacement):
    n_population = 100
    for n_samples in range(n_population + 1):
        s = sample_without_replacement(n_population, n_samples)
        assert len(s) == n_samples
        unique = np.unique(s)
        assert np.size(unique) == n_samples
        assert np.all(unique < n_population)
    assert np.size(sample_without_replacement(0, 0)) == 0