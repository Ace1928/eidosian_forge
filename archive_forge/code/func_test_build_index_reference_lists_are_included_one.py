from ... import errors, tests, transport
from .. import index as _mod_index
def test_build_index_reference_lists_are_included_one(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=1)
    builder.add_node((b'key',), b'data', ([],))
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=1\nkey\x00\x00\x00data\n\n', contents)