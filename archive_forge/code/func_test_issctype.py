import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
@pytest.mark.parametrize('rep, expected', [(np.int32, True), (list, False), (1.1, False), (str, True), (np.dtype(np.float64), True), (np.dtype((np.int16, (3, 4))), True), (np.dtype([('a', np.int8)]), True)])
def test_issctype(rep, expected):
    actual = np.issctype(rep)
    assert_equal(actual, expected)