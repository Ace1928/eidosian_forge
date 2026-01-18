import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
@pytest.mark.parametrize('numpy_ref', [True, False])
def test_array_like_fromfile(self, numpy_ref):
    self.add_method('array', self.MyArray)
    self.add_method('fromfile', self.MyArray)
    if numpy_ref is True:
        ref = np.array(1)
    else:
        ref = self.MyArray.array()
    data = np.random.random(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'testfile')
        data.tofile(fname)
        array_like = np.fromfile(fname, like=ref)
        if numpy_ref is True:
            assert type(array_like) is np.ndarray
            np_res = np.fromfile(fname, like=ref)
            assert_equal(np_res, data)
            assert_equal(array_like, np_res)
        else:
            assert type(array_like) is self.MyArray
            assert array_like.function is self.MyArray.fromfile