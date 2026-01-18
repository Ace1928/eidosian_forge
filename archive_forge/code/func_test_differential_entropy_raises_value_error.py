import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_differential_entropy_raises_value_error(self):
    random_state = np.random.RandomState(0)
    values = random_state.standard_normal((3, 100))
    error_str = 'Window length \\({window_length}\\) must be positive and less than half the sample size \\({sample_size}\\).'
    sample_size = values.shape[1]
    for window_length in {-1, 0, sample_size // 2, sample_size}:
        formatted_error_str = error_str.format(window_length=window_length, sample_size=sample_size)
        with assert_raises(ValueError, match=formatted_error_str):
            stats.differential_entropy(values, window_length=window_length, axis=1)