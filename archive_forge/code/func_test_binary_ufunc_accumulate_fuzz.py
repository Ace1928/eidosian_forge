import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_binary_ufunc_accumulate_fuzz(self):

    def get_out_axis_size(a, b, axis):
        if axis is None:
            if a.ndim == 1:
                return (a.size, False)
            else:
                return ('skip', False)
        else:
            return (a.shape[axis], False)
    self.check_unary_fuzz(np.add.accumulate, get_out_axis_size, dtype=np.int16, count=500)