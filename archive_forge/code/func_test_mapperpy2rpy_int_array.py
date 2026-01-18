import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
@pytest.mark.parametrize('ctype', ['h', 'H', 'i', 'I'])
def test_mapperpy2rpy_int_array(ctype):
    a = array.array(ctype, range(10))
    rob = robjects.default_converter.py2rpy(a)
    assert isinstance(rob, robjects.vectors.IntSexpVector)
    assert isinstance(rob, robjects.vectors.IntVector)
    assert rob.typeof == rinterface.RTYPES.INTSXP