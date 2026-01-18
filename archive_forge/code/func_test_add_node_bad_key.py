from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_node_bad_key(self):
    builder = _mod_index.GraphIndexBuilder()
    for bad_char in bytearray(b'\t\n\x0b\x0c\r\x00 '):
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'a%skey' % bytes([bad_char]),), b'data')
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (), b'data')
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, b'not-a-tuple', b'data')
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (), b'data')
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'primary', b'secondary'), b'data')
    builder = _mod_index.GraphIndexBuilder(key_elements=2)
    for bad_char in bytearray(b'\t\n\x0b\x0c\r\x00 '):
        self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'prefix', b'a%skey' % bytes([bad_char])), b'data')