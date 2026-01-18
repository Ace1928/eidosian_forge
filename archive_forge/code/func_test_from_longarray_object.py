import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_from_longarray_object():
    a = array.array('l', range(3, 103))
    vec = ri.IntSexpVector.from_object(a)
    assert tuple(range(3, 103)) == tuple(vec)