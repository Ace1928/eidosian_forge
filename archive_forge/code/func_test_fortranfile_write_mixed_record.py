import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortranfile_write_mixed_record(tmpdir):
    tf = path.join(str(tmpdir), 'test.dat')
    r1 = (('f4', 'f4', 'i4'), (np.float32(2), np.float32(3), np.int32(100)))
    r2 = (('4f4', '(3,3)f4', '8i4'), (np.random.randint(255, size=[4]).astype(np.float32), np.random.randint(255, size=[3, 3]).astype(np.float32), np.random.randint(255, size=[8]).astype(np.int32)))
    records = [r1, r2]
    for dtype, a in records:
        with FortranFile(tf, 'w') as f:
            f.write_record(*a)
        with FortranFile(tf, 'r') as f:
            b = f.read_record(*dtype)
        assert_equal(len(a), len(b))
        for aa, bb in zip(a, b):
            assert_equal(bb, aa)