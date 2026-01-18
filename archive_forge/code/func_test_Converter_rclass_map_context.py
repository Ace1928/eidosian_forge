import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_Converter_rclass_map_context():
    converter = robjects.default_converter

    class FooEnv(robjects.Environment):
        pass
    ncm = converter.rpy2py_nc_name[rinterface.SexpEnvironment]
    with robjects.default_converter.rclass_map_context(rinterface.SexpEnvironment, {'A': FooEnv}):
        assert set(ncm._map.keys()) == set('A')
    assert not len(ncm._map)