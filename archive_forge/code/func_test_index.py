import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_index():
    x = ri.IntSexpVector((1, 2, 3))
    assert x.index(1) == 0
    assert x.index(3) == 2
    with pytest.raises(ValueError):
        x.index(33)
    with pytest.raises(ValueError):
        x.index('a')