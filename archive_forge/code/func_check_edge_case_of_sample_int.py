import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb
from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement
def check_edge_case_of_sample_int(sample_without_replacement):
    with pytest.raises(ValueError):
        sample_without_replacement(0, 1)
    with pytest.raises(ValueError):
        sample_without_replacement(1, 2)
    assert sample_without_replacement(0, 0).shape == (0,)
    assert sample_without_replacement(1, 1).shape == (1,)
    assert sample_without_replacement(5, 0).shape == (0,)
    assert sample_without_replacement(5, 1).shape == (1,)
    with pytest.raises(ValueError):
        sample_without_replacement(-1, 5)
    with pytest.raises(ValueError):
        sample_without_replacement(5, -1)