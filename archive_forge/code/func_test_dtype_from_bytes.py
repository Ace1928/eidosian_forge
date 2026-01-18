import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
def test_dtype_from_bytes(self):
    assert_raises(TypeError, np.dtype, b'')
    assert_raises(TypeError, np.dtype, b'|')
    assert_dtype_equal(np.dtype(bytes([0])), np.dtype('bool'))
    assert_dtype_equal(np.dtype(bytes([17])), np.dtype(object))
    assert_dtype_equal(np.dtype(b'f'), np.dtype('float32'))
    assert_raises(TypeError, np.dtype, b'\xff')
    assert_raises(TypeError, np.dtype, b's\xff')