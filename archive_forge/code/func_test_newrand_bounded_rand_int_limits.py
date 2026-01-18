import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm._bounds import l1_min_c
from sklearn.svm._newrand import bounded_rand_int_wrap, set_seed_wrap
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('range_', [-1, _MAX_UNSIGNED_INT + 1])
def test_newrand_bounded_rand_int_limits(range_):
    """Test that `bounded_rand_int_wrap` is defined for unsigned 32bits ints"""
    with pytest.raises(OverflowError):
        bounded_rand_int_wrap(range_)