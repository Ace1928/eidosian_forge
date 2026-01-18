from ... import errors, tests, transport
from .. import index as _mod_index
def test__find_ancestors_w_missing(self):
    key1 = (b'key-1',)
    key2 = (b'key-2',)
    key3 = (b'key-3',)
    index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, b'value', ([key2],)), (key2, b'value', ([],))])
    parent_map = {}
    missing_keys = set()
    search_keys = index._find_ancestors([key2, key3], 0, parent_map, missing_keys)
    self.assertEqual({key2: ()}, parent_map)
    self.assertEqual({key3}, missing_keys)
    self.assertEqual(set(), search_keys)