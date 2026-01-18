import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('newline', ['\r', '\n', '\r\n'])
def test_manual_universal_newlines(self, newline):
    data = StringIO('0\n1\n"2\n"\n3\n4 #\n'.replace('\n', newline), newline='')
    res = np.core._multiarray_umath._load_from_filelike(data, dtype=np.dtype('U10'), filelike=True, quote='"', comment='#', skiplines=1)
    assert_array_equal(res[:, 0], ['1', f'2{newline}', '3', '4 '])