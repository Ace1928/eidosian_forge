import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test___sexp__():
    sexp = rinterface.IntSexpVector([1, 2, 3])
    sexp_count = sexp.__sexp_refcount__
    sexp_cobj = sexp.__sexp__
    d = dict(rinterface._rinterface.protected_rids())
    assert sexp_count == d[sexp.rid]
    assert sexp_count == sexp.__sexp_refcount__
    sexp2 = rinterface.IntSexpVector([4, 5, 6, 7])
    sexp2_rid = sexp2.rid
    sexp2.__sexp__ = sexp_cobj
    del sexp
    gc.collect()
    d = dict(rinterface._rinterface.protected_rids())
    assert d.get(sexp2_rid) is None