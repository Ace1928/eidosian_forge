from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_entries_reloads_and_fails(self):
    index, reload_counter = self.make_combined_index_with_missing(['1', '2', '3'])
    self.assertListRaises(transport.NoSuchFile, index.iter_all_entries)