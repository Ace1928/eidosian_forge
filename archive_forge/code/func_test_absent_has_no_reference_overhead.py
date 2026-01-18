from ... import errors, tests, transport
from .. import index as _mod_index
def test_absent_has_no_reference_overhead(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=2)
    builder.add_node((b'parent',), b'', ([(b'aail',), (b'zther',)], []))
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=2\nkey_elements=1\nlen=1\naail\x00a\x00\x00\nparent\x00\x0059\r84\t\x00\nzther\x00a\x00\x00\n\n', contents)