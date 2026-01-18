import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_default_3(self):
    assert_equal(mintypecode('fdF'), 'D')
    assert_equal(mintypecode('fdD'), 'D')
    assert_equal(mintypecode('fFD'), 'D')
    assert_equal(mintypecode('dFD'), 'D')
    assert_equal(mintypecode('ifd'), 'd')
    assert_equal(mintypecode('ifF'), 'F')
    assert_equal(mintypecode('ifD'), 'D')
    assert_equal(mintypecode('idF'), 'D')
    assert_equal(mintypecode('idD'), 'D')