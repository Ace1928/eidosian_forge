from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_entries_simple(self):
    index = self.make_index(nodes=[((b'name',), b'data')])
    self.assertEqual([(index, (b'name',), b'data')], list(index.iter_all_entries()))