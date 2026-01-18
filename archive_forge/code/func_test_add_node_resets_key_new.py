from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_add_node_resets_key_new(self):
    node = InternalNode(b'fo')
    child = LeafNode()
    child.set_maximum_size(100)
    child.map(None, (b'foo',), b'bar')
    node.add_node(b'foo', child)
    chk_bytes = self.get_chk_bytes()
    keys = list(node.serialise(chk_bytes))
    self.assertEqual(keys[1], node._key)
    node.add_node(b'fos', child)
    self.assertEqual(None, node._key)