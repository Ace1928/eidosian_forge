import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_dd_result_attributes(self):
    X = self.X
    v = self.v
    res = binned_statistic_dd(X, v, 'count', bins=3)
    attributes = ('statistic', 'bin_edges', 'binnumber')
    check_named_results(res, attributes)