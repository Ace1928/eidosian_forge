import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_mapperR2Python_s4custom(_set_class_AB):

    class A(robjects.RS4):
        pass
    sexp_a = rinterface.globalenv['A'](x=rinterface.IntSexpVector([1]))
    sexp_b = rinterface.globalenv['B'](x=rinterface.IntSexpVector([2]))
    rs4_map = conversion.get_conversion().rpy2py_nc_name[rinterface.SexpS4]
    with conversion.NameClassMapContext(rs4_map, {'A': A}):
        assert rs4_map.find_key(('A',)) == 'A'
        assert isinstance(robjects.default_converter.rpy2py(sexp_a), A)
        assert rs4_map.find_key(('B', 'A')) == 'A'
        assert isinstance(robjects.default_converter.rpy2py(sexp_b), A)