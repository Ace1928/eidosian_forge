import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_gh6308(self):
    f = getattr(self.module, self.fprefix + '_gh6308')
    assert_equal(self.module._BLNK_.name.dtype, np.dtype('S5'))
    assert_equal(len(self.module._BLNK_.name), 12)
    f('abcde', 0)
    assert_equal(self.module._BLNK_.name[0], b'abcde')
    f('12345', 5)
    assert_equal(self.module._BLNK_.name[5], b'12345')