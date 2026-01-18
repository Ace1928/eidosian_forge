from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_duplicate_key(self):
    builder = _mod_index.GraphIndexBuilder()
    builder.add_node((b'key',), b'data')
    self.assertRaises(_mod_index.BadIndexDuplicateKey, builder.add_node, (b'key',), b'data')