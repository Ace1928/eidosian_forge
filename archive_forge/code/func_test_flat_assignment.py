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
def test_flat_assignment(self):
    x = np.empty((3, 1))
    x.flat = np.arange(3)
    assert_array_almost_equal(x, [[0], [1], [2]])
    x.flat = np.arange(3, dtype=float)
    assert_array_almost_equal(x, [[0], [1], [2]])