from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__read_all_roots_multiple_old(self):
    c_map = self.make_two_deep_map()
    key1 = c_map.key()
    c_map._dump_tree()
    key1_a = c_map._root_node._items[b'a'].key()
    c_map.map((b'ccc',), b'new ccc value')
    key2 = c_map._save()
    key2_a = c_map._root_node._items[b'a'].key()
    c_map.map((b'add',), b'new add value')
    key3 = c_map._save()
    key3_a = c_map._root_node._items[b'a'].key()
    diff = self.get_difference([key3], [key1, key2], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key3], root_results)
    self.assertEqual([key1_a], diff._old_queue)
    self.assertEqual([key3_a], diff._new_queue)
    self.assertEqual([], diff._new_item_queue)