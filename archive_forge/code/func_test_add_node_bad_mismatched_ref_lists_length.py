from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_node_bad_mismatched_ref_lists_length(self):
    builder = _mod_index.GraphIndexBuilder()
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ([],))
    builder = _mod_index.GraphIndexBuilder(reference_lists=1)
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa')
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ())
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ([], []))
    builder = _mod_index.GraphIndexBuilder(reference_lists=2)
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa')
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ([],))
    self.assertRaises(_mod_index.BadIndexValue, builder.add_node, (b'akey',), b'data aa', ([], [], []))