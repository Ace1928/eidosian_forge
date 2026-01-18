import rpy2.rinterface as rinterface
from rpy2.rinterface import bufferprotocol
def test_getrank():
    v = rinterface.IntSexpVector([1, 2, 3])
    assert bufferprotocol.getrank(v.__sexp__._cdata) == 1
    m = rinterface.baseenv.find('matrix')(nrow=2, ncol=2)
    assert bufferprotocol.getrank(m.__sexp__._cdata) == 2