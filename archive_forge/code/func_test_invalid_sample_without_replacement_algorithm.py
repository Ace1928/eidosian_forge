import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb
from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement
def test_invalid_sample_without_replacement_algorithm():
    with pytest.raises(ValueError):
        sample_without_replacement(5, 4, 'unknown')