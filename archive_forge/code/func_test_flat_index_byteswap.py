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
def test_flat_index_byteswap(self):
    for dt in (np.dtype('<i4'), np.dtype('>i4')):
        x = np.array([-1, 0, 1], dtype=dt)
        assert_equal(x.flat[0].dtype, x[0].dtype)