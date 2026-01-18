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
def test_dtype_keyerrors_(self):
    dt = np.dtype([('f1', np.uint)])
    assert_raises(KeyError, dt.__getitem__, 'f2')
    assert_raises(IndexError, dt.__getitem__, 1)
    assert_raises(TypeError, dt.__getitem__, 0.0)