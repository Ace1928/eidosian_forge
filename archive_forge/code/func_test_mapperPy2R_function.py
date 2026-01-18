import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_mapperPy2R_function():
    func = lambda x: x
    rob = robjects.default_converter.py2rpy(func)
    assert isinstance(rob, robjects.SignatureTranslatedFunction)
    assert rob.typeof == rinterface.RTYPES.CLOSXP