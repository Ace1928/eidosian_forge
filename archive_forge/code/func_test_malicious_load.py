import os
import numpy as np
import tempfile
from pytest import raises as assert_raises
from numpy.testing import assert_equal, assert_
from scipy.sparse import (sparray, csc_matrix, csr_matrix, bsr_matrix, dia_matrix,
def test_malicious_load():

    class Executor:

        def __reduce__(self):
            return (assert_, (False, 'unexpected code execution'))
    fd, tmpfile = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        np.savez(tmpfile, format=Executor())
        assert_raises(ValueError, load_npz, tmpfile)
    finally:
        os.remove(tmpfile)