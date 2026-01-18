from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__read_all_roots_prepares_queues(self):
    c_map = self.make_one_deep_map(chk_map._search_key_plain)
    key1 = c_map.key()
    c_map._dump_tree()
    key1_a = c_map._root_node._items[b'a'].key()
    c_map.map((b'abb',), b'new abb content')
    key2 = c_map._save()
    key2_a = c_map._root_node._items[b'a'].key()
    diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key2], root_results)
    self.assertEqual([key2_a], diff._new_queue)
    self.assertEqual([], diff._new_item_queue)
    self.assertEqual([key1_a], diff._old_queue)