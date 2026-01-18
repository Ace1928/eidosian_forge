import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test__find_ancestors_one_parent_missing(self):
    key1 = (b'key-1',)
    key2 = (b'key-2',)
    key3 = (b'key-3',)
    index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, b'value', ([key2],)), (key2, b'value', ([key3],))])
    parent_map = {}
    missing_keys = set()
    search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
    self.assertEqual({key1: (key2,), key2: (key3,)}, parent_map)
    self.assertEqual(set(), missing_keys)
    self.assertEqual({key3}, search_keys)
    search_keys = index._find_ancestors(search_keys, 0, parent_map, missing_keys)
    self.assertEqual({key1: (key2,), key2: (key3,)}, parent_map)
    self.assertEqual({key3}, missing_keys)
    self.assertEqual(set(), search_keys)