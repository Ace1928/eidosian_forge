import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_bad_argument(self):
    self.assertRaises(ValueError, self.module._py_unhexlify, '1a')
    self.assertRaises(ValueError, self.module._py_unhexlify, b'1b')