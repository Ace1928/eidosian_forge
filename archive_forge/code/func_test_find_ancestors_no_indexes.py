from ... import errors, tests, transport
from .. import index as _mod_index
def test_find_ancestors_no_indexes(self):
    c_index = _mod_index.CombinedGraphIndex([])
    key1 = (b'key-1',)
    parent_map, missing_keys = c_index.find_ancestry([key1], 0)
    self.assertEqual({}, parent_map)
    self.assertEqual({key1}, missing_keys)