import pytest
import rpy2.rinterface as rinterface
@pytest.mark.skipif(not has_numpy, reason='Package numpy is not installed.')
def test_array_shape_len3():
    extract = rinterface.baseenv['[']
    rarray = rinterface.baseenv['array'](rinterface.IntSexpVector(range(30)), dim=rinterface.IntSexpVector([5, 2, 3]))
    npyarray = numpy.array(rarray.memoryview())
    for i in range(5):
        for j in range(2):
            for k in range(3):
                assert extract(rarray, i + 1, j + 1, k + 1)[0] == npyarray[i, j, k]