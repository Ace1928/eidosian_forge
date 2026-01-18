import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_converters_negative_indices_with_usecols():
    txt = StringIO('1.5,2.5,3.5\n3.0,4.0,XXX\n5.5,6.0,7.5\n')
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    expected = np.array([[1.5, 3.5], [3.0, np.nan], [5.5, 7.5]])
    res = np.loadtxt(txt, dtype=np.float64, delimiter=',', converters=conv, usecols=[0, -1], encoding=None)
    assert_equal(res, expected)
    res = np.loadtxt(StringIO('0,1,2\n0,1,2,3,4'), delimiter=',', usecols=[0, -1], converters={-1: lambda x: -1})
    assert_array_equal(res, [[0, -1], [0, -1]])