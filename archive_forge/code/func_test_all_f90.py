import pytest
from numpy import array
from . import util
@pytest.mark.slow
@pytest.mark.parametrize('name', 't0,t1,t2,t4,t8,s0,s1,s2,s4,s8'.split(','))
def test_all_f90(self, name):
    self.check_function(getattr(self.module.f90_return_logical, name))