import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test__find_ancestors_empty_index(self):
    index = self.make_index(ref_lists=1, key_elements=1, nodes=[])
    parent_map = {}
    missing_keys = set()
    search_keys = index._find_ancestors([('one',), ('two',)], 0, parent_map, missing_keys)
    self.assertEqual(set(), search_keys)
    self.assertEqual({}, parent_map)
    self.assertEqual({('one',), ('two',)}, missing_keys)