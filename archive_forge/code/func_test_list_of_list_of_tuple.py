import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_list_of_list_of_tuple(self):
    """Check creation from list of list of tuples"""
    h = np.array([[self._buffer]], dtype=self._descr)
    assert_(normalize_descr(self._descr) == h.dtype.descr)
    if self.multiple_rows:
        assert_(h.shape == (1, 1, 2))
    else:
        assert_(h.shape == (1, 1))