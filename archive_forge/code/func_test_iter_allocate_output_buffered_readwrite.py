import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_allocate_output_buffered_readwrite():
    a = arange(6)
    i = nditer([a, None], ['buffered', 'delay_bufalloc'], [['readonly'], ['allocate', 'readwrite']])
    with i:
        i.operands[1][:] = 1
        i.reset()
        for x in i:
            x[1][...] += x[0][...]
        assert_equal(i.operands[1], a + 1)