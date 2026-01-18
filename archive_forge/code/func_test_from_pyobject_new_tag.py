import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_from_pyobject_new_tag():
    pyobject = 'ahaha'
    sexp_new = rinterface.SexpExtPtr.from_pyobject(pyobject, tag='b')
    assert sexp_new.typeof == rinterface.RTYPES.EXTPTRSXP
    assert sexp_new.TYPE_TAG == 'b'