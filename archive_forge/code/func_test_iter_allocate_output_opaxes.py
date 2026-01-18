import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_allocate_output_opaxes():
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    i = nditer([None, a], [], [['writeonly', 'allocate'], ['readonly']], op_dtypes=[np.dtype('u4'), None], op_axes=[[1, 2, 0], None])
    assert_equal(i.operands[0].shape, (4, 2, 3))
    assert_equal(i.operands[0].strides, (4, 48, 16))
    assert_equal(i.operands[0].dtype, np.dtype('u4'))