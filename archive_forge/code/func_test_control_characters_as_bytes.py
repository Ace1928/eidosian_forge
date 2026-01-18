import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_control_characters_as_bytes():
    """Byte control characters (comments, delimiter) are supported."""
    a = np.loadtxt(StringIO('#header\n1,2,3'), comments=b'#', delimiter=b',')
    assert_equal(a, [1, 2, 3])