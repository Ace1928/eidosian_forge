import re
import unittest
from wsme import exc
from wsme import types
def test_array_eq(self):
    ell = [types.ArrayType(str)]
    assert types.ArrayType(str) in ell