import pytest
import rpy2.rinterface as rinterface
@pytest.mark.skipif(not has_numpy, reason='Package numpy is not installed.')
def test_array_struct_int():
    px = [1, -2, 3]
    x = rinterface.IntSexpVector(px)
    nx = numpy.asarray(x.memoryview())
    assert nx.dtype.kind == 'i'
    for orig, new in zip(px, nx):
        assert orig == new
    nx[1] = 12
    assert x[1] == 12