from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_entries_two_indices_dup_key(self):
    idx1 = self.make_index('name1', nodes=[((b'name',), b'data', ())])
    idx2 = self.make_index('name2', nodes=[((b'name',), b'data', ())])
    idx = _mod_index.CombinedGraphIndex([idx1, idx2])
    self.assertEqual([(idx1, (b'name',), b'data')], list(idx.iter_entries([(b'name',)])))