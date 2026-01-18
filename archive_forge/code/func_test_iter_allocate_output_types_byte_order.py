import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_allocate_output_types_byte_order():
    a = array([3], dtype='u4').newbyteorder()
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']])
    assert_equal(i.dtypes[0], i.dtypes[1])
    i = nditer([a, a, None], [], [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_(i.dtypes[0] != i.dtypes[2])
    assert_equal(i.dtypes[0].newbyteorder('='), i.dtypes[2])