import pytest
import rpy2.rinterface as rinterface
@pytest.mark.skip(reason='WIP')
@pytest.mark.skipif(not has_numpy, reason='Package numpy is not installed.')
def test_array_struct_complex():
    px = [1 + 2j, 2 + 5j, -1 + 0j]
    x = rinterface.ComplexSexpVector(px)
    nx = numpy.asarray(x.memoryview())
    assert nx.dtype.kind == 'c'
    for orig, new in zip(px, nx):
        assert orig == new