import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('data', [['1,2\n', '2,3\r\n'], ['1,2\n', "'2\n',3\n"], ['1,2\n', "'2\r',3\n"], ['1,2\n', "'2\r\n',3\n"]])
def test_good_newline_in_iterator(data):
    res = np.loadtxt(data, delimiter=',', quotechar="'")
    assert_array_equal(res, [[1.0, 2.0], [2.0, 3.0]])