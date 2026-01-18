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
def test_bool_flat_indexing_invalid_nr_elements(self):
    s = np.ones(10, dtype=float)
    x = np.array((15,), dtype=float)

    def ia(x, s, v):
        x[s > 0] = v
    assert_raises(IndexError, ia, x, s, np.zeros(9, dtype=float))
    assert_raises(IndexError, ia, x, s, np.zeros(11, dtype=float))
    assert_raises(ValueError, ia, x.flat, s, np.zeros(9, dtype=float))
    assert_raises(ValueError, ia, x.flat, s, np.zeros(11, dtype=float))