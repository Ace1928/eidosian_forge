import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
def test_ignore_shape_range():
    msg = 'No generator is defined for the shape parameters'
    with pytest.raises(ValueError, match=msg):
        rng = FastGeneratorInversion(stats.t(0.03))
    rng = FastGeneratorInversion(stats.t(0.03), ignore_shape_range=True)
    u_err, _ = rng.evaluate_error(size=1000, random_state=234)
    assert u_err >= 1e-06