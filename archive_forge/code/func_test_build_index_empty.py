from ... import errors, tests, transport
from .. import index as _mod_index
def test_build_index_empty(self):
    builder = _mod_index.GraphIndexBuilder()
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=1\nlen=0\n\n', contents)