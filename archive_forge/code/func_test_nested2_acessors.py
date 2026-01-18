import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_nested2_acessors(self):
    """Check reading the nested fields of a nested array (2nd level)"""
    h = np.array(self._buffer, dtype=self._descr)
    if not self.multiple_rows:
        assert_equal(h['Info']['Info2']['value'], np.array(self._buffer[1][2][1], dtype='c16'))
        assert_equal(h['Info']['Info2']['z3'], np.array(self._buffer[1][2][3], dtype='u4'))
    else:
        assert_equal(h['Info']['Info2']['value'], np.array([self._buffer[0][1][2][1], self._buffer[1][1][2][1]], dtype='c16'))
        assert_equal(h['Info']['Info2']['z3'], np.array([self._buffer[0][1][2][3], self._buffer[1][1][2][3]], dtype='u4'))