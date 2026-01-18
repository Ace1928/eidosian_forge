import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_do_slot_reassign():
    sexp = rinterface.IntSexpVector([])
    slot_value_a = rinterface.IntSexpVector([3])
    sexp.do_slot_assign('foo', slot_value_a)
    slot_value_b = rinterface.IntSexpVector([5, 6])
    sexp.do_slot_assign('foo', slot_value_b)
    slot_value_back = sexp.do_slot('foo')
    assert len(slot_value_b) == len(slot_value_back)
    assert all((x == y for x, y in zip(slot_value_b, slot_value_back)))