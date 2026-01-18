import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortran_eof_broken_record(tmpdir):
    filename = path.join(str(tmpdir), 'scratch')
    np.random.seed(1)
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    with open(filename, 'ab') as f:
        f.truncate(path.getsize(filename) - 20)
    with FortranFile(filename, 'r') as f:
        assert len(f.read_reals()) == 5
        with pytest.raises(FortranFormattingError):
            f.read_reals()