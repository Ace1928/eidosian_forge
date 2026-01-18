import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_endian_bool_indexing(self):
    a = np.arange(10.0, dtype='>f8')
    b = np.arange(10.0, dtype='<f8')
    xa = np.where((a > 2) & (a < 6))
    xb = np.where((b > 2) & (b < 6))
    ya = (a > 2) & (a < 6)
    yb = (b > 2) & (b < 6)
    assert_array_almost_equal(xa, ya.nonzero())
    assert_array_almost_equal(xb, yb.nonzero())
    assert_(np.all(a[ya] > 0.5))
    assert_(np.all(b[yb] > 0.5))