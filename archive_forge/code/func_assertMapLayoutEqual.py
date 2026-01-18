from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def assertMapLayoutEqual(self, map_one, map_two):
    """Assert that the internal structure is identical between the maps."""
    map_one._ensure_root()
    node_one_stack = [map_one._root_node]
    map_two._ensure_root()
    node_two_stack = [map_two._root_node]
    while node_one_stack:
        node_one = node_one_stack.pop()
        node_two = node_two_stack.pop()
        if node_one.__class__ != node_two.__class__:
            self.assertEqualDiff(map_one._dump_tree(include_keys=True), map_two._dump_tree(include_keys=True))
        self.assertEqual(node_one._search_prefix, node_two._search_prefix)
        if isinstance(node_one, InternalNode):
            self.assertEqual(sorted(node_one._items.keys()), sorted(node_two._items.keys()))
            node_one_stack.extend(sorted([n for n, _ in node_one._iter_nodes(map_one._store)], key=lambda a: a._search_prefix))
            node_two_stack.extend(sorted([n for n, _ in node_two._iter_nodes(map_two._store)], key=lambda a: a._search_prefix))
        else:
            self.assertEqual(node_one._items, node_two._items)
    self.assertEqual([], node_two_stack)