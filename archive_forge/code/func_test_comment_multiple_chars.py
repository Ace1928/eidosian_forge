import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('comment', ['..', '//', '@-', 'this is a comment:'])
def test_comment_multiple_chars(comment):
    content = '# IGNORE\n1.5, 2.5# ABC\n3.0,4.0# XXX\n5.5,6.0\n'
    txt = StringIO(content.replace('#', comment))
    a = np.loadtxt(txt, delimiter=',', comments=comment)
    assert_equal(a, [[1.5, 2.5], [3.0, 4.0], [5.5, 6.0]])