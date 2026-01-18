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
@pytest.mark.parametrize('dtype', list(np.typecodes['All']) + [rational])
def test_basic_dtypes_subclass_properties(self, dtype):
    dtype = np.dtype(dtype)
    assert isinstance(dtype, np.dtype)
    assert type(dtype) is not np.dtype
    if dtype.type.__name__ != 'rational':
        dt_name = type(dtype).__name__.lower().removesuffix('dtype')
        if dt_name == 'uint' or dt_name == 'int':
            dt_name += 'c'
        sc_name = dtype.type.__name__
        assert dt_name == sc_name.strip('_')
        assert type(dtype).__module__ == 'numpy.dtypes'
        assert getattr(numpy.dtypes, type(dtype).__name__) is type(dtype)
    else:
        assert type(dtype).__name__ == 'dtype[rational]'
        assert type(dtype).__module__ == 'numpy'
    assert not type(dtype)._abstract
    parametric = (np.void, np.str_, np.bytes_, np.datetime64, np.timedelta64)
    if dtype.type not in parametric:
        assert not type(dtype)._parametric
        assert type(dtype)() is dtype
    else:
        assert type(dtype)._parametric
        with assert_raises(TypeError):
            type(dtype)()