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
def test_creating_dtype_with_dtype_class_errors():
    with pytest.raises(TypeError, match='Cannot convert np.dtype into a'):
        np.array(np.ones(10), dtype=np.dtype)