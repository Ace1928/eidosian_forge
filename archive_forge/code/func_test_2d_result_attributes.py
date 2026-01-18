import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_2d_result_attributes(self):
    x = self.x
    y = self.y
    v = self.v
    res = binned_statistic_2d(x, y, v, 'count', bins=5)
    attributes = ('statistic', 'x_edge', 'y_edge', 'binnumber')
    check_named_results(res, attributes)