import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_differential_entropy_vasicek(self):
    random_state = np.random.RandomState(0)
    values = random_state.standard_normal(100)
    entropy = stats.differential_entropy(values, method='vasicek')
    assert_allclose(entropy, 1.342551, rtol=1e-06)
    entropy = stats.differential_entropy(values, window_length=1, method='vasicek')
    assert_allclose(entropy, 1.122044, rtol=1e-06)
    entropy = stats.differential_entropy(values, window_length=8, method='vasicek')
    assert_allclose(entropy, 1.349401, rtol=1e-06)