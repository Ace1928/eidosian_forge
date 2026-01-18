import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_character_assignment(self):
    arr = np.zeros((1, 5), dtype='c')
    arr[0] = np.str_('asdfg')
    assert_array_equal(arr[0], np.array('asdfg', dtype='c'))
    assert arr[0, 1] == b's'