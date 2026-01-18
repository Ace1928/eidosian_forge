import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_charsxp_encoding():
    encoding = rinterface.NA_Character.encoding
    assert encoding == rinterface.sexp.CETYPE.CE_NATIVE