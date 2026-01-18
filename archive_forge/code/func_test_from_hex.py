import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_from_hex(self):
    self.assertUnhexlify(b'0123456789abcdef0123456789abcdef01234567')
    self.assertUnhexlify(b'123456789abcdef0123456789abcdef012345678')
    self.assertUnhexlify(b'0123456789ABCDEF0123456789ABCDEF01234567')
    self.assertUnhexlify(b'123456789ABCDEF0123456789ABCDEF012345678')
    hex_chars = binascii.hexlify(bytes(range(256)))
    for i in range(0, 480, 40):
        self.assertUnhexlify(hex_chars[i:i + 40])
    self.assertUnhexlify(hex_chars[480:] + hex_chars[0:8])