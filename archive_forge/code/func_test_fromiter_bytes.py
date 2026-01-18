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
def test_fromiter_bytes(self):
    a = np.fromiter(list(range(10)), dtype='b')
    b = np.fromiter(list(range(10)), dtype='B')
    assert_(np.all(a == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    assert_(np.all(b == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))