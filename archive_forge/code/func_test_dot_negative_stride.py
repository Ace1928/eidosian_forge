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
def test_dot_negative_stride(self):
    x = np.array([[1, 5, 25, 125.0, 625]])
    y = np.array([[20.0], [160.0], [640.0], [1280.0], [1024.0]])
    z = y[::-1].copy()
    y2 = y[::-1]
    assert_equal(np.dot(x, z), np.dot(x, y2))