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
def test_arange_non_native_dtype(self):
    for T in ('>f4', '<f4'):
        dt = np.dtype(T)
        assert_equal(np.arange(0, dtype=dt).dtype, dt)
        assert_equal(np.arange(0.5, dtype=dt).dtype, dt)
        assert_equal(np.arange(5, dtype=dt).dtype, dt)