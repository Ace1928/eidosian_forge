import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('nl', ('\n', '\r'))
def test_control_character_newline_raises(nl):
    txt = StringIO(f'1{nl}2{nl}3{nl}{nl}4{nl}5{nl}6{nl}{nl}')
    msg = 'control character.*cannot be a newline'
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, delimiter=nl)
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, comments=nl)
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, quotechar=nl)