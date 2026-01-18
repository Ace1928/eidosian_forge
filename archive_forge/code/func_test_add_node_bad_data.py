from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_node_bad_data(self):
    builder = _mod_index.GraphIndexBuilder()
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data\naa')
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data\x00aa')