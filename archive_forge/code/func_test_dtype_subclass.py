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
@pytest.mark.parametrize('code', np.typecodes['All'])
def test_dtype_subclass(self, code: str) -> None:
    cls = type(np.dtype(code))
    alias = cls[Any]
    assert isinstance(alias, types.GenericAlias)
    assert alias.__origin__ is cls