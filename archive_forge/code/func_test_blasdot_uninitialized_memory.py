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
@_no_tracing
def test_blasdot_uninitialized_memory(self):
    for m in [0, 1, 2]:
        for n in [0, 1, 2]:
            for k in range(3):
                x = np.array([1.23456789e+207], dtype=np.float64)
                if IS_PYPY:
                    x.resize((m, 0), refcheck=False)
                else:
                    x.resize((m, 0))
                y = np.array([1.23456789e+207], dtype=np.float64)
                if IS_PYPY:
                    y.resize((0, n), refcheck=False)
                else:
                    y.resize((0, n))
                z = np.dot(x, y)
                assert_(np.all(z == 0))
                assert_(z.shape == (m, n))