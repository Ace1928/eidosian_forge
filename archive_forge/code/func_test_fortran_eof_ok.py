import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortran_eof_ok(tmpdir):
    filename = path.join(str(tmpdir), 'scratch')
    np.random.seed(1)
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    with FortranFile(filename, 'r') as f:
        assert len(f.read_reals()) == 5
        assert len(f.read_reals()) == 3
        with pytest.raises(FortranEOFError):
            f.read_reals()