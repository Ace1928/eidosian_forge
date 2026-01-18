import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_invalid_not_sha1(self):
    self.assertKeyToSha1(None, (_hex_form,))
    self.assertKeyToSha1(None, (b'sha2:' + _hex_form,))