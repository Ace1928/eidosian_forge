import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test__sexp__wrongtypeof():
    sexp = rinterface.IntSexpVector([1, 2, 3])
    cobj = sexp.__sexp__
    sexp = rinterface.StrSexpVector(['a', 'b'])
    assert len(sexp) == 2
    with pytest.raises(ValueError):
        sexp.__sexp__ = cobj