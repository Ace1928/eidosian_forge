import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_gh3425(self):
    f = getattr(self.module, self.fprefix + '_gh3425')
    assert_equal(f('abC'), b'ABC')
    assert_equal(f(''), b'')
    assert_equal(f('abC12d'), b'ABC12D')