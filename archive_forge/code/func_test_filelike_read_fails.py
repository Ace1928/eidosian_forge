import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_filelike_read_fails(self):

    class BadFileLike:
        counter = 0

        def read(self, size):
            self.counter += 1
            if self.counter > 20:
                raise RuntimeError('Bad bad bad!')
            return '1,2,3\n'
    with pytest.raises(RuntimeError, match='Bad bad bad!'):
        np.core._multiarray_umath._load_from_filelike(BadFileLike(), dtype=np.dtype('i'), filelike=True)