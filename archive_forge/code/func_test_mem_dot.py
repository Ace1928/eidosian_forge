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
def test_mem_dot(self):
    x = np.random.randn(0, 1)
    y = np.random.randn(10, 1)
    _z = np.ones(10)
    _dummy = np.empty((0, 10))
    z = np.lib.stride_tricks.as_strided(_z, _dummy.shape, _dummy.strides)
    np.dot(x, np.transpose(y), out=z)
    assert_equal(_z, np.ones(10))
    np.core.multiarray.dot(x, np.transpose(y), out=z)
    assert_equal(_z, np.ones(10))