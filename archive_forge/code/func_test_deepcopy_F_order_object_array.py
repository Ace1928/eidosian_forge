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
def test_deepcopy_F_order_object_array(self):
    a = {'a': 1}
    b = {'b': 2}
    arr = np.array([[a, b], [a, b]], order='F')
    arr_cp = copy.deepcopy(arr)
    assert_equal(arr, arr_cp)
    assert_(arr is not arr_cp)
    assert_(arr[0, 1] is not arr_cp[1, 1])
    assert_(arr[0, 1] is arr[1, 1])
    assert_(arr_cp[0, 1] is arr_cp[1, 1])