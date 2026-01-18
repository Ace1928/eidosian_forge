import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_init_from_seq_invalid_item():
    seq = (1, 'b', 3)
    with pytest.raises(ValueError):
        ri.IntSexpVector(seq)