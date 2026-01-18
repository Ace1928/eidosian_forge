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
def test_field_order_equality(self):
    x = np.dtype({'names': ['A', 'B'], 'formats': ['i4', 'f4'], 'offsets': [0, 4]})
    y = np.dtype({'names': ['B', 'A'], 'formats': ['i4', 'f4'], 'offsets': [4, 0]})
    assert_equal(x == y, False)
    assert np.can_cast(x, y, casting='safe')