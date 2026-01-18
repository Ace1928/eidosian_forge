import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_invalid_not_string(self):
    self.assertKeyToSha1(None, (None,))
    self.assertKeyToSha1(None, (list(_hex_form),))