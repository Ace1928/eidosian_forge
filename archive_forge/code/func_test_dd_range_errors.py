import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_dd_range_errors(self):
    with assert_raises(ValueError, match='In range, start must be <= stop'):
        binned_statistic_dd([self.y], self.v, range=[[1, 0]])
    with assert_raises(ValueError, match='In dimension 1 of range, start must be <= stop'):
        binned_statistic_dd([self.x, self.y], self.v, range=[[1, 0], [0, 1]])
    with assert_raises(ValueError, match='In dimension 2 of range, start must be <= stop'):
        binned_statistic_dd([self.x, self.y], self.v, range=[[0, 1], [1, 0]])
    with assert_raises(ValueError, match='range given for 1 dimensions; 2 required'):
        binned_statistic_dd([self.x, self.y], self.v, range=[[0, 1]])