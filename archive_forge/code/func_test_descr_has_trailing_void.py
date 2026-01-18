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
def test_descr_has_trailing_void(self):
    dtype = np.dtype({'names': ['A', 'B'], 'formats': ['f4', 'f4'], 'offsets': [0, 8], 'itemsize': 16})
    new_dtype = np.dtype(dtype.descr)
    assert_equal(new_dtype.itemsize, 16)