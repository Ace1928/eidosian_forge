import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
def test_evaluate_error_inputs():
    gen = FastGeneratorInversion(stats.norm())
    with pytest.raises(ValueError, match='size must be an integer'):
        gen.evaluate_error(size=3.5)
    with pytest.raises(ValueError, match='size must be an integer'):
        gen.evaluate_error(size=(3, 3))