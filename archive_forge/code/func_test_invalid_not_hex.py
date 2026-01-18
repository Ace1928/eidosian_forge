import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_invalid_not_hex(self):
    self.assertKeyToSha1(None, (b'sha1:abcdefghijklmnopqrstuvwxyz12345678901234',))