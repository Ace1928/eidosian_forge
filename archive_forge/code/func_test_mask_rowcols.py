import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_mask_rowcols(self):
    x = array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert_equal(mask_rowcols(x).mask, [[1, 1, 1], [1, 0, 0], [1, 0, 0]])
    assert_equal(mask_rowcols(x, 0).mask, [[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert_equal(mask_rowcols(x, 1).mask, [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    x = array(x._data, mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert_equal(mask_rowcols(x).mask, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    assert_equal(mask_rowcols(x, 0).mask, [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    assert_equal(mask_rowcols(x, 1).mask, [[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert_equal(mask_rowcols(x).mask, [[1, 1, 1], [1, 1, 1], [1, 1, 0]])
    assert_equal(mask_rowcols(x, 0).mask, [[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    assert_equal(mask_rowcols(x, 1).mask, [[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert_(mask_rowcols(x).all() is masked)
    assert_(mask_rowcols(x, 0).all() is masked)
    assert_(mask_rowcols(x, 1).all() is masked)
    assert_(mask_rowcols(x).mask.all())
    assert_(mask_rowcols(x, 0).mask.all())
    assert_(mask_rowcols(x, 1).mask.all())