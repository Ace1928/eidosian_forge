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
def test_metadata_rejects_nondict(self):
    assert_raises(TypeError, np.dtype, int, metadata='datum')
    assert_raises(TypeError, np.dtype, int, metadata=1)
    assert_raises(TypeError, np.dtype, int, metadata=None)