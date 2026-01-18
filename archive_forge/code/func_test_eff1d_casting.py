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
def test_eff1d_casting(self):
    x = np.array([1, 2, 4, 7, 0], dtype=np.int16)
    res = np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
    assert_equal(res, [-99, 1, 2, 3, -7, 88, 99])
    res = np.ediff1d(x, to_begin=1 << 20, to_end=1 << 20)
    assert_equal(res, [0, 1, 2, 3, -7, 0])