import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def assertHexlify(self, as_binary):
    self.assertEqual(binascii.hexlify(as_binary), self.module._py_hexlify(as_binary))