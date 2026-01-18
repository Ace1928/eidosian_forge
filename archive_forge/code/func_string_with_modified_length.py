import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
def string_with_modified_length(self, dtype, change_length):
    fact = 1 if dtype.char == 'S' else 4
    length = dtype.itemsize // fact + change_length
    return np.dtype(f'{dtype.byteorder}{dtype.char}{length}')