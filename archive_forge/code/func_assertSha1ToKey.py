import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def assertSha1ToKey(self, hex_sha1):
    bin_sha1 = binascii.unhexlify(hex_sha1)
    key = self.module._py_sha1_to_key(bin_sha1)
    self.assertEqual((b'sha1:' + hex_sha1,), key)