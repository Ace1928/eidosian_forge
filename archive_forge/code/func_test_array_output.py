import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_array_output(self):
    f = getattr(self.module, self.fprefix + '_array_output')
    assert_array_equal(f(list(map(ord, 'abc'))), np.array(list('abc'), dtype='S1'))