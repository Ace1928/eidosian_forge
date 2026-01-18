import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.special import xlogy
from scipy.stats.contingency import (margins, expected_freq,
def test_bad_association_args():
    assert_raises(ValueError, association, [[1, 2], [3, 4]], 'X')
    assert_raises(ValueError, association, [[[1, 2]], [[3, 4]]], 'cramer')
    assert_raises(ValueError, association, [[-1, 10], [1, 2]], 'cramer')
    assert_raises(ValueError, association, np.array([[1, 2], ['dd', 4]], dtype=object), 'cramer')