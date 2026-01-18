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
def test_assign_from_sequence_error(self):
    arr = np.array([1, 2, 3])
    assert_raises(ValueError, arr.__setitem__, slice(None), [9, 9])
    arr.__setitem__(slice(None), [9])
    assert_equal(arr, [9, 9, 9])