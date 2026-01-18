from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_add_node_empty_new(self):
    node = InternalNode(b'fo')
    child = LeafNode()
    child.set_maximum_size(100)
    child.map(None, (b'foo',), b'bar')
    node.add_node(b'foo', child)
    self.assertEqual(3, node._node_width)
    self.assertEqual({(b'foo',): b'bar'}, self.to_dict(node, None))
    self.assertEqual(1, len(node))
    chk_bytes = self.get_chk_bytes()
    keys = list(node.serialise(chk_bytes))
    child_key = child.serialise(chk_bytes)[0]
    self.assertEqual([child_key, (b'sha1:cf67e9997d8228a907c1f5bfb25a8bd9cd916fac',)], keys)
    bytes = self.read_bytes(chk_bytes, keys[1])
    node = chk_map._deserialise(bytes, keys[1], None)
    self.assertEqual(1, len(node))
    self.assertEqual({(b'foo',): b'bar'}, self.to_dict(node, chk_bytes))
    self.assertEqual(3, node._node_width)