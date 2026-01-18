import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_gh5927(self):
    x = self.x
    v = self.v
    statistics = ['mean', 'median', 'count', 'sum']
    for statistic in statistics:
        binned_statistic(x, v, statistic, bins=10)