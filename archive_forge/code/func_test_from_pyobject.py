import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_from_pyobject():
    pyobject = 'ahaha'
    sexp_new = rinterface.SexpExtPtr.from_pyobject(pyobject)
    assert rinterface.RTYPES.EXTPTRSXP == sexp_new.typeof