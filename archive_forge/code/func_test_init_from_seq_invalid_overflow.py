import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
@pytest.mark.skip(reason='WIP')
@pytest.mark.skipif(struct.calcsize('P') < 8, reason='Only relevant on 64 architectures.')
def test_init_from_seq_invalid_overflow():
    MAX_INT = ri._rinterface._MAX_INT
    v = ri.IntSexpVector((MAX_INT, 42))
    assert v[0] == MAX_INT
    assert v[1] == 42
    with pytest.raises(OverflowError):
        ri.IntSexpVector((MAX_INT + 1,))