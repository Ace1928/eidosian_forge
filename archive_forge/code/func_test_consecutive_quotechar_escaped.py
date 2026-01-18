import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_consecutive_quotechar_escaped():
    txt = StringIO('"Hello, my name is ""Monty""!"')
    expected = np.array('Hello, my name is "Monty"!', dtype='U40')
    res = np.loadtxt(txt, dtype='U40', delimiter=',', quotechar='"')
    assert_equal(res, expected)