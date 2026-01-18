from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_entries_prefix_reloads(self):
    index, reload_counter = self.make_combined_index_with_missing()
    result = list(index.iter_entries_prefix([(b'1',)]))
    index3 = index._indices[0]
    self.assertEqual([(index3, (b'1',), b'')], result)
    self.assertEqual([1, 1, 0], reload_counter)