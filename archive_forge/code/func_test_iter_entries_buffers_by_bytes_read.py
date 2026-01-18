from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_entries_buffers_by_bytes_read(self):
    index = self.make_index(nodes=self.make_nodes(64))
    list(index.iter_entries([self.make_key(10)]))
    self.assertIs(None, index._nodes)
    self.assertEqual(4096, index._bytes_read)
    list(index.iter_entries([self.make_key(11)]))
    self.assertIs(None, index._nodes)
    self.assertEqual(4096, index._bytes_read)
    list(index.iter_entries([self.make_key(40)]))
    self.assertIs(None, index._nodes)
    self.assertEqual(8192, index._bytes_read)
    list(index.iter_entries([self.make_key(32)]))
    self.assertIs(None, index._nodes)
    list(index.iter_entries([self.make_key(60)]))
    self.assertIsNot(None, index._nodes)