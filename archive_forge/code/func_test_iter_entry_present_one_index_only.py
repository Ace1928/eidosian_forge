from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_entry_present_one_index_only(self):
    idx1 = self.make_index('1', nodes=[((b'key',), b'', ())])
    idx2 = self.make_index('2', nodes=[])
    idx = _mod_index.CombinedGraphIndex([idx1, idx2])
    self.assertEqual([(idx1, (b'key',), b'')], list(idx.iter_entries([(b'key',)])))
    idx = _mod_index.CombinedGraphIndex([idx2, idx1])
    self.assertEqual([(idx1, (b'key',), b'')], list(idx.iter_entries([(b'key',)])))