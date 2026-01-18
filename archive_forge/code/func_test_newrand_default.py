import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm._bounds import l1_min_c
from sklearn.svm._newrand import bounded_rand_int_wrap, set_seed_wrap
from sklearn.utils.fixes import CSR_CONTAINERS
def test_newrand_default():
    """Test that bounded_rand_int_wrap without seeding respects the range

    Note this test should pass either if executed alone, or in conjunctions
    with other tests that call set_seed explicit in any order: it checks
    invariants on the RNG instead of specific values.
    """
    generated = [bounded_rand_int_wrap(100) for _ in range(10)]
    assert all((0 <= x < 100 for x in generated))
    assert not all((x == generated[0] for x in generated))