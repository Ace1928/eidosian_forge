from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__read_all_roots_different_depths(self):
    c_map = self.make_two_deep_map(chk_map._search_key_plain)
    c_map._dump_tree()
    key1 = c_map.key()
    key1_a = c_map._root_node._items[b'a'].key()
    key1_c = c_map._root_node._items[b'c'].key()
    key1_d = c_map._root_node._items[b'd'].key()
    c_map2 = self.make_one_deep_two_prefix_map(chk_map._search_key_plain)
    c_map2._dump_tree()
    key2 = c_map2.key()
    key2_aa = c_map2._root_node._items[b'aa'].key()
    key2_ad = c_map2._root_node._items[b'ad'].key()
    diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key2], root_results)
    self.assertEqual([key1_a], diff._old_queue)
    self.assertEqual({key2_aa, key2_ad}, set(diff._new_queue))
    self.assertEqual([], diff._new_item_queue)
    diff = self.get_difference([key1], [key2], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key1], root_results)
    self.assertEqual({key2_aa, key2_ad}, set(diff._old_queue))
    self.assertEqual({key1_a, key1_c, key1_d}, set(diff._new_queue))
    self.assertEqual([], diff._new_item_queue)