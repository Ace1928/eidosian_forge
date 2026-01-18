import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('iterable, batch, expected', batch_generator_cases)
def test_batch_generator(self, iterable, batch, expected):
    got = list(_resampling._batch_generator(iterable, batch))
    assert got == expected