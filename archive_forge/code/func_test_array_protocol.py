import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_array_protocol():
    v = ri.IntSexpVector(range(10))
    ai = v.__array_interface__
    assert ai['version'] == 3