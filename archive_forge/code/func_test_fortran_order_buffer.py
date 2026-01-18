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
def test_fortran_order_buffer(self):
    import numpy as np
    a = np.array([['Hello', 'Foob']], dtype='U5', order='F')
    arr = np.ndarray(shape=[1, 2, 5], dtype='U1', buffer=a)
    arr2 = np.array([[['H', 'e', 'l', 'l', 'o'], ['F', 'o', 'o', 'b', '']]])
    assert_array_equal(arr, arr2)