import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('ws', (' ', '\t', '\u2003', '\xa0', '\u3000'))
def test_blank_lines_spaces_delimit(ws):
    txt = StringIO(f'1 2{ws}30\n\n{ws}\n4 5 60{ws}\n  {ws}  \n7 8 {ws} 90\n  # comment\n3 2 1')
    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])
    assert_equal(np.loadtxt(txt, dtype=int, delimiter=None, comments='#'), expected)