import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_getitem_outofbound():
    x = ri.IntSexpVector((1, 2, 3))
    with pytest.raises(IndexError):
        x.__getitem__(10)