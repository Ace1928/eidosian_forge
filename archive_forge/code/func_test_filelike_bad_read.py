import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_filelike_bad_read(self):

    class BadFileLike:
        counter = 0

        def read(self, size):
            return 1234
    with pytest.raises(TypeError, match='non-string returned while reading data'):
        np.core._multiarray_umath._load_from_filelike(BadFileLike(), dtype=np.dtype('i'), filelike=True)