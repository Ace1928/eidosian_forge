import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
def simple_dtype_instances():
    for dtype_class in simple_dtypes:
        dt = dtype_class()
        yield pytest.param(dt, id=str(dt))
        if dt.byteorder != '|':
            dt = dt.newbyteorder()
            yield pytest.param(dt, id=str(dt))