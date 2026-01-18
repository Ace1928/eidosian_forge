import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_entropy_zero(self):
    assert_almost_equal(stats.entropy([0, 1, 2]), 0.6365141682948128, decimal=12)