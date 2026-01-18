from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__process_next_old_batched_no_dupes(self):
    c_map = self.make_two_deep_map()
    key1 = c_map.key()
    c_map._dump_tree()
    key1_a = c_map._root_node._items[b'a'].key()
    key1_aa = c_map._root_node._items[b'a']._items[b'aa'].key()
    key1_ab = c_map._root_node._items[b'a']._items[b'ab'].key()
    key1_ac = c_map._root_node._items[b'a']._items[b'ac'].key()
    key1_ad = c_map._root_node._items[b'a']._items[b'ad'].key()
    c_map.map((b'aaa',), b'new aaa value')
    key2 = c_map._save()
    key2_a = c_map._root_node._items[b'a'].key()
    key2_aa = c_map._root_node._items[b'a']._items[b'aa'].key()
    c_map.map((b'acc',), b'new acc content')
    key3 = c_map._save()
    key3_a = c_map._root_node._items[b'a'].key()
    key3_ac = c_map._root_node._items[b'a']._items[b'ac'].key()
    diff = self.get_difference([key3], [key1, key2], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key3], root_results)
    self.assertEqual(sorted([key1_a, key2_a]), sorted(diff._old_queue))
    self.assertEqual([key3_a], diff._new_queue)
    self.assertEqual([], diff._new_item_queue)
    diff._process_next_old()
    self.assertEqual(sorted([key1_aa, key1_ab, key1_ac, key1_ad, key2_aa]), sorted(diff._old_queue))