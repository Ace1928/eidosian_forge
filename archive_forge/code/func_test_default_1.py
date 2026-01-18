import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_default_1(self):
    for itype in '1bcsuwil':
        assert_equal(mintypecode(itype), 'd')
    assert_equal(mintypecode('f'), 'f')
    assert_equal(mintypecode('d'), 'd')
    assert_equal(mintypecode('F'), 'F')
    assert_equal(mintypecode('D'), 'D')