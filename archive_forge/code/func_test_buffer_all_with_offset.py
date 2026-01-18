from ... import errors, tests, transport
from .. import index as _mod_index
def test_buffer_all_with_offset(self):
    nodes = self.make_nodes(200)
    idx = self.make_index_with_offset(offset=1234567, nodes=nodes)
    idx._buffer_all()
    self.assertEqual(200, idx.key_count())