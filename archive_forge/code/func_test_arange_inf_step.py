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
def test_arange_inf_step(self):
    ref = np.arange(0, 1, 10)
    x = np.arange(0, 1, np.inf)
    assert_array_equal(ref, x)
    ref = np.arange(0, 1, -10)
    x = np.arange(0, 1, -np.inf)
    assert_array_equal(ref, x)
    ref = np.arange(0, -1, -10)
    x = np.arange(0, -1, -np.inf)
    assert_array_equal(ref, x)
    ref = np.arange(0, -1, 10)
    x = np.arange(0, -1, np.inf)
    assert_array_equal(ref, x)