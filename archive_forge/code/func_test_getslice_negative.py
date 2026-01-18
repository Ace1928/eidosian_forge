import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_getslice_negative():
    vec = ri.IntSexpVector([1, 2, 3])
    vec_s = vec[-2:-1]
    assert len(vec_s) == 1
    assert vec_s[0] == 2