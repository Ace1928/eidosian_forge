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
def test_assign_obj_listoflists(self):
    a = np.zeros(4, dtype=object)
    b = a.copy()
    a[0] = [1]
    a[1] = [2]
    a[2] = [3]
    a[3] = [4]
    b[...] = [[1], [2], [3], [4]]
    assert_equal(a, b)
    a = np.zeros((2, 2), dtype=object)
    a[...] = [[1, 2]]
    assert_equal(a, [[1, 2], [1, 2]])