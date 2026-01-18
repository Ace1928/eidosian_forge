from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_duplicate_key_2_elements(self):
    builder = _mod_index.GraphIndexBuilder(key_elements=2)
    builder.add_node((b'key', b'key'), b'data')
    self.assertRaises(_mod_index.BadIndexDuplicateKey, builder.add_node, (b'key', b'key'), b'data')