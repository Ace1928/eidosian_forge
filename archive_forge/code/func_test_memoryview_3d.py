import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
@pytest.mark.skipif(not has_numpy, reason='numpy currently required for memoryview to work.')
def test_memoryview_3d():
    shape = (5, 2, 3)
    values = tuple(range(30))
    rarray = ri.baseenv['array'](ri.IntSexpVector(values), dim=ri.IntSexpVector(shape))
    mv = rarray.memoryview()
    assert mv.f_contiguous is True
    assert mv.shape == shape
    assert tuple((x[0][0] for x in mv.tolist())) == (0, 1, 2, 3, 4)