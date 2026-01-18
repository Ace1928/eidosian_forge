from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__read_all_roots_skips_known_old(self):
    c_map = self.make_one_deep_map(chk_map._search_key_plain)
    key1 = c_map.key()
    c_map2 = self.make_root_only_map(chk_map._search_key_plain)
    key2 = c_map2.key()
    diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([], root_results)