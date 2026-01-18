import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
def test_deprecated_scalar(self, dtype):
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)

    def scalar(value, dtype):
        dtype.type(value)

    def assign(value, dtype):
        arr = np.array([0, 0, 0], dtype=dtype)
        arr[2] = value

    def create(value, dtype):
        np.array([value], dtype=dtype)
    for creation_func in [scalar, assign, create]:
        try:
            self.assert_deprecated(lambda: creation_func(info.min - 1, dtype))
        except OverflowError:
            pass
        try:
            self.assert_deprecated(lambda: creation_func(info.max + 1, dtype))
        except OverflowError:
            pass