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
def test_broadcast_flat_assignment(self):
    x = np.empty((3, 1))

    def bfa():
        x[:] = np.arange(3)

    def bfb():
        x[:] = np.arange(3, dtype=float)
    assert_raises(ValueError, bfa)
    assert_raises(ValueError, bfb)