import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_access_top_fields(self):
    """Check reading the top fields of a nested array"""
    h = np.array(self._buffer, dtype=self._descr)
    if not self.multiple_rows:
        assert_(h.shape == ())
        assert_equal(h['x'], np.array(self._buffer[0], dtype='i4'))
        assert_equal(h['y'], np.array(self._buffer[4], dtype='f8'))
        assert_equal(h['z'], np.array(self._buffer[5], dtype='u1'))
    else:
        assert_(len(h) == 2)
        assert_equal(h['x'], np.array([self._buffer[0][0], self._buffer[1][0]], dtype='i4'))
        assert_equal(h['y'], np.array([self._buffer[0][4], self._buffer[1][4]], dtype='f8'))
        assert_equal(h['z'], np.array([self._buffer[0][5], self._buffer[1][5]], dtype='u1'))