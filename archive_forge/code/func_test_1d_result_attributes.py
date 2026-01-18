import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_1d_result_attributes(self):
    x = self.x
    v = self.v
    res = binned_statistic(x, v, 'count', bins=10)
    attributes = ('statistic', 'bin_edges', 'binnumber')
    check_named_results(res, attributes)