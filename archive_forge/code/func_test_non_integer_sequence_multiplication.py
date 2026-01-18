import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_non_integer_sequence_multiplication(self):

    def mult(a, b):
        return a * b
    assert_raises(TypeError, mult, [1], np.float_(3))
    mult([1], np.int_(3))