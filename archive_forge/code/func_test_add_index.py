from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_index(self):
    idx = _mod_index.CombinedGraphIndex([])
    idx1 = self.make_index('name', 0, nodes=[((b'key',), b'', ())])
    idx.insert_index(0, idx1)
    self.assertEqual([(idx1, (b'key',), b'')], list(idx.iter_all_entries()))