import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
@pytest.mark.parametrize('dt', [dt1, dt2, dt3, dt4, dt5, dt6])
def test_load_padded_dtype(tmpdir, dt):
    arr = np.zeros(3, dt)
    for i in range(3):
        arr[i] = i + 5
    npz_file = os.path.join(tmpdir, 'aligned.npz')
    np.savez(npz_file, arr=arr)
    with np.load(npz_file) as npz:
        arr1 = npz['arr']
    assert_array_equal(arr, arr1)