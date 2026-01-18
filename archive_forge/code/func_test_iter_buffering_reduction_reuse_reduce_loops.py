import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffering_reduction_reuse_reduce_loops():
    a = np.zeros((2, 7))
    b = np.zeros((1, 7))
    it = np.nditer([a, b], flags=['reduce_ok', 'external_loop', 'buffered'], op_flags=[['readonly'], ['readwrite']], buffersize=5)
    with it:
        bufsizes = [x.shape[0] for x, y in it]
    assert_equal(bufsizes, [5, 2, 5, 2])
    assert_equal(sum(bufsizes), a.size)