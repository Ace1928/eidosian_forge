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
def test_equivalent_record(self):
    """Test whether equivalent subarray dtypes hash the same."""
    a = np.dtype((int, (2, 3)))
    b = np.dtype((int, (2, 3)))
    assert_dtype_equal(a, b)