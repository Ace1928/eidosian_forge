import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_reduction_error():
    a = np.arange(6)
    assert_raises(ValueError, nditer, [a, None], [], [['readonly'], ['readwrite', 'allocate']], op_axes=[[0], [-1]])
    a = np.arange(6).reshape(2, 3)
    assert_raises(ValueError, nditer, [a, None], ['external_loop'], [['readonly'], ['readwrite', 'allocate']], op_axes=[[0, 1], [-1, -1]])