import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb
from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement
def check_sample_int_distribution(sample_without_replacement):
    n_population = 10
    n_trials = 10000
    for n_samples in range(n_population):
        n_expected = comb(n_population, n_samples, exact=True)
        output = {}
        for i in range(n_trials):
            output[frozenset(sample_without_replacement(n_population, n_samples))] = None
            if len(output) == n_expected:
                break
        else:
            raise AssertionError('number of combinations != number of expected (%s != %s)' % (len(output), n_expected))