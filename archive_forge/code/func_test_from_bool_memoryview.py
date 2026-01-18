import array
import pytest
import rpy2.rinterface as ri
def test_from_bool_memoryview():
    a = array.array('b', (True, False, True))
    mv = memoryview(a)
    with pytest.raises(ValueError):
        ri.BoolSexpVector.from_memoryview(mv)