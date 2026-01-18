import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_allocate_output_types_scalar():
    i = nditer([None, 1, 2.3, np.float32(12), np.complex128(3)], [], [['writeonly', 'allocate']] + [['readonly']] * 4)
    assert_equal(i.operands[0].dtype, np.dtype('complex128'))
    assert_equal(i.operands[0].ndim, 0)