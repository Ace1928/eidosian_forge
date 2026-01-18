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
def iter_struct_object_dtypes():
    """
    Iterates over a few complex dtypes and object pattern which
    fill the array with a given object (defaults to a singleton).

    Yields
    ------
    dtype : dtype
    pattern : tuple
        Structured tuple for use with `np.array`.
    count : int
        Number of objects stored in the dtype.
    singleton : object
        A singleton object. The returned pattern is constructed so that
        all objects inside the datatype are set to the singleton.
    """
    obj = object()
    dt = np.dtype([('b', 'O', (2, 3))])
    p = ([[obj] * 3] * 2,)
    yield pytest.param(dt, p, 6, obj, id='<subarray>')
    dt = np.dtype([('a', 'i4'), ('b', 'O', (2, 3))])
    p = (0, [[obj] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id='<subarray in field>')
    dt = np.dtype([('a', 'i4'), ('b', [('ba', 'O'), ('bb', 'i1')], (2, 3))])
    p = (0, [[(obj, 0)] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id='<structured subarray 1>')
    dt = np.dtype([('a', 'i4'), ('b', [('ba', 'O'), ('bb', 'O')], (2, 3))])
    p = (0, [[(obj, obj)] * 3] * 2)
    yield pytest.param(dt, p, 12, obj, id='<structured subarray 2>')