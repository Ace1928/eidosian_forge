import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_do_slot_assign_empty_string():
    sexp = rinterface.IntSexpVector([])
    slot_value = rinterface.IntSexpVector([3])
    with pytest.raises(ValueError):
        sexp.do_slot_assign('', slot_value)