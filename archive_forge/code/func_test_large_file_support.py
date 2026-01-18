import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_large_file_support(tmpdir):
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        pytest.skip('Unknown if Windows has sparse filesystems')
    tf_name = os.path.join(tmpdir, 'sparse_file')
    try:
        import subprocess as sp
        sp.check_call(['truncate', '-s', '5368709120', tf_name])
    except Exception:
        pytest.skip('Could not create 5GB large file')
    with open(tf_name, 'wb') as f:
        f.seek(5368709120)
        d = np.arange(5)
        np.save(f, d)
    with open(tf_name, 'rb') as f:
        f.seek(5368709120)
        r = np.load(f)
    assert_array_equal(r, d)