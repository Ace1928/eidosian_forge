from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_entries_reloads_midway(self):
    index, reload_counter = self.make_combined_index_with_missing(['2'])
    index1, index2 = index._indices
    result = list(index.iter_entries([(b'1',), (b'2',), (b'3',)]))
    index3 = index._indices[0]
    self.assertEqual([(index1, (b'1',), b''), (index3, (b'2',), b'')], result)
    self.assertEqual([1, 1, 0], reload_counter)