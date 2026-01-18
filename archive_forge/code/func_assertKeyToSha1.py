import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def assertKeyToSha1(self, expected, key):
    if expected is None:
        expected_bin = None
    else:
        expected_bin = binascii.unhexlify(expected)
    actual_sha1 = self.module._py_key_to_sha1(key)
    if expected_bin != actual_sha1:
        actual_hex_sha1 = None
        if actual_sha1 is not None:
            actual_hex_sha1 = binascii.hexlify(actual_sha1)
        self.fail('_key_to_sha1 returned:\n    %s\n != %s' % (actual_sha1, expected))