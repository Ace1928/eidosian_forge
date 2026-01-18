import array
import pytest
import rpy2.rinterface as ri
def test_from_int_memoryview():
    a = array.array('i', (True, False, True))
    mv = memoryview(a)
    vec = ri.BoolSexpVector.from_memoryview(mv)
    assert (True, False, True) == tuple(vec)