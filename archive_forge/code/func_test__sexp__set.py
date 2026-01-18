import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test__sexp__set():
    x = rinterface.IntSexpVector([1, 2, 3])
    x_s = x.__sexp__
    x_rid = x.rid
    assert x.__sexp_refcount__ == 1
    y = rinterface.IntSexpVector([4, 5, 6])
    y_count = y.__sexp_refcount__
    y_rid = y.rid
    assert y_count == 1
    assert x_rid in [elt[0] for elt in rinterface._rinterface.protected_rids()]
    x.__sexp__ = y.__sexp__
    assert x_rid in [elt[0] for elt in rinterface._rinterface.protected_rids()]
    del x_s
    assert x_rid not in [elt[0] for elt in rinterface._rinterface.protected_rids()]
    assert x.rid == y.rid
    assert y_rid == y.rid