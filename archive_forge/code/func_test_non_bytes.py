import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_non_bytes(self):
    self.assertInvalid('type=leaf\n')