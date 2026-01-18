import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_comment_quotechar_collision_raises():
    with pytest.raises(TypeError, match='.*control characters.*incompatible'):
        np.loadtxt(StringIO('1 2 3'), comments='#', quotechar='#')