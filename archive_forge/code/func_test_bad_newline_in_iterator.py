import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('data', [['1,2\n', '2\n,3\n'], ['1,2\n', '2\r,3\n']])
def test_bad_newline_in_iterator(data):
    msg = 'Found an unquoted embedded newline within a single line'
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(data, delimiter=',')