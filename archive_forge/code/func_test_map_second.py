from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_second(self):
    node = LeafNode()
    node.map(None, (b'foo bar',), b'baz quux')
    result = node.map(None, (b'bingo',), b'bango')
    self.assertEqual((b'', [(b'', node)]), result)
    self.assertEqual({(b'foo bar',): b'baz quux', (b'bingo',): b'bango'}, self.to_dict(node, None))
    self.assertEqual(2, len(node))