import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_error_message_2(self):
    """Check the message is formatted correctly when either x or y is a scalar."""
    x = 2
    y = np.ones(20)
    with pytest.raises(AssertionError) as exc_info:
        self._assert_func(x, y)
    msgs = str(exc_info.value).split('\n')
    assert_equal(msgs[3], 'Mismatched elements: 20 / 20 (100%)')
    assert_equal(msgs[4], 'Max absolute difference: 1.')
    assert_equal(msgs[5], 'Max relative difference: 1.')
    y = 2
    x = np.ones(20)
    with pytest.raises(AssertionError) as exc_info:
        self._assert_func(x, y)
    msgs = str(exc_info.value).split('\n')
    assert_equal(msgs[3], 'Mismatched elements: 20 / 20 (100%)')
    assert_equal(msgs[4], 'Max absolute difference: 1.')
    assert_equal(msgs[5], 'Max relative difference: 0.5')