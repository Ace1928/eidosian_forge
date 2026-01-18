from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_entries_no_reload(self):
    index, reload_counter = self.make_combined_index_with_missing()
    index._reload_func = None
    self.assertListRaises(transport.NoSuchFile, index.iter_all_entries)