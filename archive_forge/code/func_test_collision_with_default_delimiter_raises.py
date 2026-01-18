import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('ws', (' ', '\t', '\u2003', '\xa0', '\u3000'))
def test_collision_with_default_delimiter_raises(ws):
    with pytest.raises(TypeError, match='.*control characters.*incompatible'):
        np.loadtxt(StringIO(f'1{ws}2{ws}3\n4{ws}5{ws}6\n'), comments=ws)
    with pytest.raises(TypeError, match='.*control characters.*incompatible'):
        np.loadtxt(StringIO(f'1{ws}2{ws}3\n4{ws}5{ws}6\n'), quotechar=ws)