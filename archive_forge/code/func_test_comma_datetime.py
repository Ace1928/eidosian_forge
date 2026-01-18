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
def test_comma_datetime(self):
    dt = np.dtype('M8[D],datetime64[Y],i8')
    assert_equal(dt, np.dtype([('f0', 'M8[D]'), ('f1', 'datetime64[Y]'), ('f2', 'i8')]))