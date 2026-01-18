import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
@pytest.mark.parametrize('index', [(..., [1, 2], slice(None)), ([0, 1], ..., 0), (..., [1, 2], [1, 2])])
def test_broadcast_error_reports_correct_shape(self, index):
    values = np.zeros((100, 100))
    arr = np.zeros((3, 4, 5, 6, 7))
    shape_str = str(arr[index].shape).replace(' ', '')
    with pytest.raises(ValueError) as e:
        arr[index] = values
    assert str(e.value).endswith(shape_str)