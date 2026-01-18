import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.skipif('win32' in sys.platform or numpy.intp(0).itemsize < 8, reason='do not run on 32 bit or windows (no sparse memory)')
def test_map_coordinates_large_data(self):
    try:
        n = 30000
        a = numpy.empty(n ** 2, dtype=numpy.float32).reshape(n, n)
        a[n - 3:, n - 3:] = 0
        ndimage.map_coordinates(a, [[n - 1.5], [n - 1.5]], order=1)
    except MemoryError as e:
        raise pytest.skip('Not enough memory available') from e