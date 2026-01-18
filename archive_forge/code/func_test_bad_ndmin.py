import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('badval', [-1, 3, None, 'plate of shrimp'])
def test_bad_ndmin(badval):
    with pytest.raises(ValueError, match='Illegal value of ndmin keyword'):
        np.loadtxt('foo.bar', ndmin=badval)