from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__read_all_roots_multiple_targets(self):
    c_map = self.make_root_only_map()
    key1 = c_map.key()
    c_map = self.make_one_deep_map()
    key2 = c_map.key()
    c_map._dump_tree()
    key2_c = c_map._root_node._items[b'c'].key()
    key2_d = c_map._root_node._items[b'd'].key()
    c_map.map((b'ccc',), b'new ccc value')
    key3 = c_map._save()
    key3_c = c_map._root_node._items[b'c'].key()
    diff = self.get_difference([key2, key3], [key1], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual(sorted([key2, key3]), sorted(root_results))
    self.assertEqual([], diff._old_queue)
    self.assertEqual(sorted([key2_c, key3_c, key2_d]), sorted(diff._new_queue))
    self.assertEqual([], diff._new_item_queue)