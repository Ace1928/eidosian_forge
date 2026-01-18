import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_getslice_missingboundary():
    vec = ri.IntSexpVector(range(1, 11))
    vec_slice = vec[:2]
    assert len(vec_slice) == 2
    assert vec_slice[0] == 1
    assert vec_slice[1] == 2
    vec_slice = vec[8:]
    assert len(vec_slice) == 2
    assert vec_slice[0] == 9
    assert vec_slice[1] == 10
    vec_slice = vec[-2:]
    assert len(vec_slice) == 2
    assert vec_slice[0] == 9
    assert vec_slice[1] == 10