import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
@pytest.mark.parametrize('ctype', ['d', 'f'])
def test_mapperpy2rpy_float_array(ctype):
    a = array.array(ctype, (1.1, 2.2, 3.3))
    rob = robjects.default_converter.py2rpy(a)
    assert isinstance(rob, robjects.vectors.FloatSexpVector)
    assert isinstance(rob, robjects.vectors.FloatVector)
    assert rob.typeof == rinterface.RTYPES.REALSXP