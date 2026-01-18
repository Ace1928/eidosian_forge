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
def test_byteswap_complex_scalar(self):
    for dtype in [np.dtype('<' + t) for t in np.typecodes['Complex']]:
        z = np.array([2.2 - 1.1j], dtype)
        x = z[0]
        y = x.byteswap()
        if x.dtype.byteorder == z.dtype.byteorder:
            assert_equal(x, np.frombuffer(y.tobytes(), dtype=dtype.newbyteorder()))
        else:
            assert_equal(x, np.frombuffer(y.tobytes(), dtype=dtype))
        assert_equal(x.real, y.real.byteswap())
        assert_equal(x.imag, y.imag.byteswap())