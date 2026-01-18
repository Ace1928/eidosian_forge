import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_names():
    sexp = rinterface.baseenv.find('.Platform')
    names = sexp.names
    assert len(names) > 1
    assert 'OS.type' in names