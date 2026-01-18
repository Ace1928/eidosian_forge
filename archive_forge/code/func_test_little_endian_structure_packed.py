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
def test_little_endian_structure_packed(self):

    class LittleEndStruct(ctypes.LittleEndianStructure):
        _fields_ = [('one', ctypes.c_uint8), ('two', ctypes.c_uint32)]
        _pack_ = 1
    expected = np.dtype([('one', 'u1'), ('two', '<u4')])
    self.check(LittleEndStruct, expected)