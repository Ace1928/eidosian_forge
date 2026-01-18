from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_missing_entry_empty_no_size(self):
    idx = self.make_index()
    idx = _mod_index.GraphIndex(idx._transport, 'index', None)
    self.assertEqual([], list(idx.iter_entries([(b'a',)])))