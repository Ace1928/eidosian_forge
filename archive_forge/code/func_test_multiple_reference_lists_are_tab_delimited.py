from ... import errors, tests, transport
from .. import index as _mod_index
def test_multiple_reference_lists_are_tab_delimited(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=2)
    builder.add_node((b'keference',), b'data', ([], []))
    builder.add_node((b'rey',), b'data', ([(b'keference',)], [(b'keference',)]))
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=2\nkey_elements=1\nlen=2\nkeference\x00\x00\t\x00data\nrey\x00\x0059\t59\x00data\n\n', contents)