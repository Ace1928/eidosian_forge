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
def test_ndmin_float64(self):
    x = np.array([1, 2, 3], dtype=np.float64)
    assert_equal(np.array(x, dtype=np.float32, ndmin=2).ndim, 2)
    assert_equal(np.array(x, dtype=np.float64, ndmin=2).ndim, 2)