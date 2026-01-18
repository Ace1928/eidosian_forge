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
def test_from_dict_with_zero_width_field(self):
    dt = np.dtype([('val1', np.float32, (0,)), ('val2', int)])
    dt2 = np.dtype({'names': ['val1', 'val2'], 'formats': [(np.float32, (0,)), int]})
    assert_dtype_equal(dt, dt2)
    assert_equal(dt.fields['val1'][0].itemsize, 0)
    assert_equal(dt.itemsize, dt.fields['val2'][0].itemsize)