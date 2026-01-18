import pytest
from numpy.core._simd import targets
@pytest.mark.skipif(not npyv2, reason='could not find a second SIMD extension with NPYV support')
def test_nomix(self):
    a = npyv.setall_u32(1)
    a2 = npyv2.setall_u32(1)
    pytest.raises(TypeError, npyv.add_u32, a2, a2)
    pytest.raises(TypeError, npyv2.add_u32, a, a)