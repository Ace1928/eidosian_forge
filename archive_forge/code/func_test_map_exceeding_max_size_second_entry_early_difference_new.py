from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_exceeding_max_size_second_entry_early_difference_new(self):
    node = LeafNode()
    node.set_maximum_size(10)
    node.map(None, (b'foo bar',), b'baz quux')
    prefix, result = list(node.map(None, (b'blue',), b'red'))
    self.assertEqual(b'', prefix)
    self.assertEqual(2, len(result))
    split_chars = {result[0][0], result[1][0]}
    self.assertEqual({b'f', b'b'}, split_chars)
    nodes = dict(result)
    node = nodes[b'f']
    self.assertEqual({(b'foo bar',): b'baz quux'}, self.to_dict(node, None))
    self.assertEqual(10, node.maximum_size)
    self.assertEqual(1, node._key_width)
    node = nodes[b'b']
    self.assertEqual({(b'blue',): b'red'}, self.to_dict(node, None))
    self.assertEqual(10, node.maximum_size)
    self.assertEqual(1, node._key_width)