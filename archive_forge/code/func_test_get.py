import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_get():
    sexp = rinterface.baseenv.find('letters')
    assert sexp.typeof == rinterface.RTYPES.STRSXP
    sexp = rinterface.baseenv.find('pi')
    assert sexp.typeof == rinterface.RTYPES.REALSXP
    sexp = rinterface.baseenv.find('options')
    assert sexp.typeof == rinterface.RTYPES.CLOSXP