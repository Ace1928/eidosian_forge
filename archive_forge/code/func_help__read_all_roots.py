from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def help__read_all_roots(self, search_key_func):
    c_map = self.make_root_only_map(search_key_func=search_key_func)
    key1 = c_map.key()
    c_map.map((b'aaa',), b'new aaa content')
    key2 = c_map._save()
    diff = self.get_difference([key2], [key1], search_key_func)
    root_results = [record.key for record in diff._read_all_roots()]
    self.assertEqual([key2], root_results)
    self.assertEqual([((b'aaa',), b'new aaa content')], diff._new_item_queue)
    self.assertEqual([], diff._new_queue)
    self.assertEqual([], diff._old_queue)