from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__read_all_roots_mixed_depth(self):
    c_map = self.make_one_deep_two_prefix_map(chk_map._search_key_plain)
    c_map._dump_tree()
    key1 = c_map.key()
    key1_aa = c_map._root_node._items[b'aa'].key()
    key1_ad = c_map._root_node._items[b'ad'].key()
    c_map2 = self.make_one_deep_one_prefix_map(chk_map._search_key_plain)
    c_map2._dump_tree()
    key2 = c_map2.key()
    key2_a = c_map2._root_node._items[b'a'].key()
    key2_b = c_map2._root_node._items[b'b'].key()
    diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key2], root_results)
    self.assertEqual([], diff._old_queue)
    self.assertEqual([key2_b], diff._new_queue)
    self.assertEqual([], diff._new_item_queue)
    diff = self.get_difference([key1], [key2], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key1], root_results)
    self.assertEqual([key2_a], diff._old_queue)
    self.assertEqual([key1_aa], diff._new_queue)
    self.assertEqual([], diff._new_item_queue)