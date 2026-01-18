import array
import pytest
import rpy2.rinterface as ri
def test_init_from_seq():
    seq = (1.0, 2.0, 3.0)
    v = ri.FloatSexpVector(seq)
    assert len(v) == 3
    for x, y in zip(seq, v):
        assert x == y