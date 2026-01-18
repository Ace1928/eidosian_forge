import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_complex64_alignment(self):
    dtt = np.complex64
    arr = np.arange(10, dtype=dtt)
    arr2 = np.reshape(arr, (2, 5))
    data_str = arr2.tobytes('F')
    data_back = np.ndarray(arr2.shape, arr2.dtype, buffer=data_str, order='F')
    assert_array_equal(arr2, data_back)