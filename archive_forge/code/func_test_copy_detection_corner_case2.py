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
@pytest.mark.skipif(np.ones(1).strides[0] == np.iinfo(np.intp).max, reason='Using relaxed stride debug')
def test_copy_detection_corner_case2(self):
    b = np.indices((0, 3, 4)).T.reshape(-1, 3)
    assert_equal(b.strides, (3 * b.itemsize, b.itemsize))