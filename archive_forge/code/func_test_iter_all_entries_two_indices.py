from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_entries_two_indices(self):
    idx1 = self.make_index('name1', nodes=[((b'name',), b'data', ())])
    idx2 = self.make_index('name2', nodes=[((b'2',), b'', ())])
    idx = _mod_index.CombinedGraphIndex([idx1, idx2])
    self.assertEqual([(idx1, (b'name',), b'data'), (idx2, (b'2',), b'')], list(idx.iter_all_entries()))