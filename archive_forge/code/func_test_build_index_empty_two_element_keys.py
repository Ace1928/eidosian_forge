from ... import errors, tests, transport
from .. import index as _mod_index
def test_build_index_empty_two_element_keys(self):
    builder = _mod_index.GraphIndexBuilder(key_elements=2)
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=2\nlen=0\n\n', contents)