import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_nested2_descriptor(self):
    """Check access nested descriptors of a nested array (2nd level)"""
    h = np.array(self._buffer, dtype=self._descr)
    assert_(h.dtype['Info']['Info2']['value'].name == 'void256')
    assert_(h.dtype['Info']['Info2']['z3'].name == 'void64')