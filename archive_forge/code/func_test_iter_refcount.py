import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_iter_refcount():
    a = arange(6)
    dt = np.dtype('f4').newbyteorder()
    rc_a = sys.getrefcount(a)
    rc_dt = sys.getrefcount(dt)
    with nditer(a, [], [['readwrite', 'updateifcopy']], casting='unsafe', op_dtypes=[dt]) as it:
        assert_(not it.iterationneedsapi)
        assert_(sys.getrefcount(a) > rc_a)
        assert_(sys.getrefcount(dt) > rc_dt)
    it = None
    assert_equal(sys.getrefcount(a), rc_a)
    assert_equal(sys.getrefcount(dt), rc_dt)
    a = arange(6, dtype='f4')
    dt = np.dtype('f4')
    rc_a = sys.getrefcount(a)
    rc_dt = sys.getrefcount(dt)
    it = nditer(a, [], [['readwrite']], op_dtypes=[dt])
    rc2_a = sys.getrefcount(a)
    rc2_dt = sys.getrefcount(dt)
    it2 = it.copy()
    assert_(sys.getrefcount(a) > rc2_a)
    assert_(sys.getrefcount(dt) > rc2_dt)
    it = None
    assert_equal(sys.getrefcount(a), rc2_a)
    assert_equal(sys.getrefcount(dt), rc2_dt)
    it2 = None
    assert_equal(sys.getrefcount(a), rc_a)
    assert_equal(sys.getrefcount(dt), rc_dt)
    del it2