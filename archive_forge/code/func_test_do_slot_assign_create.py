import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_do_slot_assign_create():
    sexp = rinterface.IntSexpVector([])
    slot_value = rinterface.IntSexpVector([3])
    sexp.do_slot_assign('foo', slot_value)
    slot_value_back = sexp.do_slot('foo')
    assert len(slot_value_back) == len(slot_value)
    assert all((x == y for x, y in zip(slot_value, slot_value_back)))