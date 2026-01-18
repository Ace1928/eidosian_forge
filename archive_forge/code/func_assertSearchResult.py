from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def assertSearchResult(self, start_keys, stop_keys, key_count, parent_map, missing_keys, tip_keys, depth):
    start, stop, count = vf_search.limited_search_result_from_parent_map(parent_map, missing_keys, tip_keys, depth)
    self.assertEqual((sorted(start_keys), sorted(stop_keys), key_count), (sorted(start), sorted(stop), count))