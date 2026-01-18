import re
import unittest
from wsme import exc
from wsme import types
def test_binary_from_base(self):
    import base64
    assert types.binary.frombasetype(None) is None
    encoded = base64.encodebytes(b'abcdef')
    assert types.binary.frombasetype(encoded) == b'abcdef'