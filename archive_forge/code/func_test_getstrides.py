import rpy2.rinterface as rinterface
from rpy2.rinterface import bufferprotocol
def test_getstrides():
    v = rinterface.IntSexpVector([1, 2, 3])
    assert bufferprotocol.getstrides(v.__sexp__._cdata, [3], 8) == (8,)
    m = rinterface.baseenv.find('matrix')(nrow=2, ncol=3)
    shape = (2, 3)
    sizeof = 8
    expected = (sizeof, shape[0] * sizeof)
    assert bufferprotocol.getstrides(m.__sexp__._cdata, shape, sizeof) == expected