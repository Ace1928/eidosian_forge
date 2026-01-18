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
@pytest.mark.parametrize(['type_char', 'char_size', 'scalar_type'], [['U', 4, np.str_], ['S', 1, np.bytes_]])
def test_create_string_dtypes_directly(self, type_char, char_size, scalar_type):
    dtype_class = type(np.dtype(type_char))
    dtype = dtype_class(8)
    assert dtype.type is scalar_type
    assert dtype.itemsize == 8 * char_size