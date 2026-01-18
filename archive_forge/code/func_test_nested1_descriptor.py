import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_nested1_descriptor(self):
    """Check access nested descriptors of a nested array (1st level)"""
    h = np.array(self._buffer, dtype=self._descr)
    assert_(h.dtype['Info']['value'].name == 'complex128')
    assert_(h.dtype['Info']['y2'].name == 'float64')
    assert_(h.dtype['info']['Name'].name == 'str256')
    assert_(h.dtype['info']['Value'].name == 'complex128')