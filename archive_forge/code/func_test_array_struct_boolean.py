import pytest
import rpy2.rinterface as rinterface
@pytest.mark.skipif(not has_numpy, reason='Package numpy is not installed.')
def test_array_struct_boolean():
    px = [True, False, True]
    x = rinterface.BoolSexpVector(px)
    nx = numpy.asarray(x.memoryview())
    assert nx.dtype.kind == 'i'
    for orig, new in zip(px, nx):
        assert orig == new