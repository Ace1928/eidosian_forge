import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_masked_constant(self):
    actual = mr_[np.ma.masked, 1]
    assert_equal(actual.mask, [True, False])
    assert_equal(actual.data[1], 1)
    actual = mr_[[1, 2], np.ma.masked]
    assert_equal(actual.mask, [False, False, True])
    assert_equal(actual.data[:2], [1, 2])