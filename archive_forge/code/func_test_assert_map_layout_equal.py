from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_assert_map_layout_equal(self):
    store = self.get_chk_bytes()
    map_one = CHKMap(store, None)
    map_one._root_node.set_maximum_size(20)
    map_two = CHKMap(store, None)
    map_two._root_node.set_maximum_size(20)
    self.assertMapLayoutEqual(map_one, map_two)
    map_one.map((b'aaa',), b'value')
    self.assertRaises(AssertionError, self.assertMapLayoutEqual, map_one, map_two)
    map_two.map((b'aaa',), b'value')
    self.assertMapLayoutEqual(map_one, map_two)
    map_one.map((b'aab',), b'value')
    self.assertIsInstance(map_one._root_node, InternalNode)
    self.assertRaises(AssertionError, self.assertMapLayoutEqual, map_one, map_two)
    map_two.map((b'aab',), b'value')
    self.assertMapLayoutEqual(map_one, map_two)
    map_one.map((b'aac',), b'value')
    self.assertRaises(AssertionError, self.assertMapLayoutEqual, map_one, map_two)
    self.assertCanonicalForm(map_one)