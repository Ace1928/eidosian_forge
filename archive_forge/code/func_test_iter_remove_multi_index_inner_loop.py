import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_remove_multi_index_inner_loop():
    a = arange(24).reshape(2, 3, 4)
    i = nditer(a, ['multi_index'])
    assert_equal(i.ndim, 3)
    assert_equal(i.shape, (2, 3, 4))
    assert_equal(i.itviews[0].shape, (2, 3, 4))
    before = [x for x in i]
    i.remove_multi_index()
    after = [x for x in i]
    assert_equal(before, after)
    assert_equal(i.ndim, 1)
    assert_raises(ValueError, lambda i: i.shape, i)
    assert_equal(i.itviews[0].shape, (24,))
    i.reset()
    assert_equal(i.itersize, 24)
    assert_equal(i[0].shape, tuple())
    i.enable_external_loop()
    assert_equal(i.itersize, 24)
    assert_equal(i[0].shape, (24,))
    assert_equal(i.value, arange(24))