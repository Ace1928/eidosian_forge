from ... import errors, tests, transport
from .. import index as _mod_index
def test_key_count_reloads_and_fails(self):
    idx, reload_counter = self.make_combined_index_with_missing(['1', '2', '3'])
    self.assertRaises(transport.NoSuchFile, idx.key_count)
    self.assertEqual([2, 1, 1], reload_counter)