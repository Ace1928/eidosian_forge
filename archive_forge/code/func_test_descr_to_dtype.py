import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
@pytest.mark.parametrize('dt', [np.dtype(np.dtype([('a', np.int8), ('b', np.int16), ('c', np.int32)], align=True), (3,)), np.dtype([('x', np.dtype({'names': ['a', 'b'], 'formats': ['i1', 'i1'], 'offsets': [0, 4], 'itemsize': 8}, (3,)), (4,))]), np.dtype([('x', ('<f8', (5,)), (2,))]), np.dtype([('x', np.dtype((np.dtype((np.dtype({'names': ['a', 'b'], 'formats': ['i1', 'i1'], 'offsets': [0, 4], 'itemsize': 8}), (3,))), (4,))))]), np.dtype([('a', np.dtype((np.dtype((np.dtype((np.dtype([('a', int), ('b', np.dtype({'names': ['a', 'b'], 'formats': ['i1', 'i1'], 'offsets': [0, 4], 'itemsize': 8}))]), (3,))), (4,))), (5,))))])])
def test_descr_to_dtype(dt):
    dt1 = format.descr_to_dtype(dt.descr)
    assert_equal_(dt1, dt)
    arr1 = np.zeros(3, dt)
    arr2 = roundtrip(arr1)
    assert_array_equal(arr1, arr2)