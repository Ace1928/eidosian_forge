import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
@pytest.mark.parametrize('axis', [None, 0, 1])
@pytest.mark.parametrize(['func', 'rowcols_axis'], [(np.ma.mask_rows, 0), (np.ma.mask_cols, 1)])
def test_mask_row_cols_axis_deprecation(self, axis, func, rowcols_axis):
    x = array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    with assert_warns(DeprecationWarning):
        res = func(x, axis=axis)
        assert_equal(res, mask_rowcols(x, rowcols_axis))