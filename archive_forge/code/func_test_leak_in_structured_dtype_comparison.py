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
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_leak_in_structured_dtype_comparison(self):
    recordtype = np.dtype([('a', np.float64), ('b', np.int32), ('d', (str, 5))])
    a = np.zeros(2, dtype=recordtype)
    for i in range(100):
        a == a
    assert_(sys.getrefcount(a) < 10)
    before = sys.getrefcount(a)
    u, v = (a[0], a[1])
    u == v
    del u, v
    gc.collect()
    after = sys.getrefcount(a)
    assert_equal(before, after)