import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_blank_lines_normal_delimiter():
    txt = StringIO('1,2,30\n\n4,5,60\n\n7,8,90\n# comment\n3,2,1')
    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])
    assert_equal(np.loadtxt(txt, dtype=int, delimiter=',', comments='#'), expected)