from ... import errors, tests, transport
from .. import index as _mod_index
def test__find_ancestors(self):
    key1 = (b'key-1',)
    key2 = (b'key-2',)
    index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, b'value', ([key2],)), (key2, b'value', ([],))])
    parent_map = {}
    missing_keys = set()
    search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
    self.assertEqual({key1: (key2,)}, parent_map)
    self.assertEqual(set(), missing_keys)
    self.assertEqual({key2}, search_keys)
    search_keys = index._find_ancestors(search_keys, 0, parent_map, missing_keys)
    self.assertEqual({key1: (key2,), key2: ()}, parent_map)
    self.assertEqual(set(), missing_keys)
    self.assertEqual(set(), search_keys)