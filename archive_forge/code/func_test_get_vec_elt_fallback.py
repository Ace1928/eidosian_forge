import pytest
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface
@pytest.mark.parametrize('rcls,value,func,fallback', ((rpy2.rinterface.IntSexpVector, [1, 2, 3], openrlib.INTEGER_ELT, openrlib._get_integer_elt_fallback), (rpy2.rinterface.BoolSexpVector, [True, True, False], openrlib.LOGICAL_ELT, openrlib._get_logical_elt_fallback), (rpy2.rinterface.FloatSexpVector, [1.1, 2.2, 3.3], openrlib.REAL_ELT, openrlib._get_real_elt_fallback)))
def test_get_vec_elt_fallback(rcls, value, func, fallback):
    rpy2.rinterface.initr()
    v = rcls(value)
    assert func(v.__sexp__._cdata, 1) == fallback(v.__sexp__._cdata, 1)