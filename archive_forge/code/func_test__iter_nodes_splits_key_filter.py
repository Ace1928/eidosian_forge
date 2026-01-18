from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__iter_nodes_splits_key_filter(self):
    node = InternalNode(b'')
    child = LeafNode()
    child.set_maximum_size(100)
    child.map(None, (b'foo',), b'bar')
    node.add_node(b'f', child)
    child = LeafNode()
    child.set_maximum_size(100)
    child.map(None, (b'bar',), b'baz')
    node.add_node(b'b', child)
    key_filter = ((b'foo',), (b'bar',), (b'cat',))
    for child, node_key_filter in node._iter_nodes(None, key_filter=key_filter):
        self.assertEqual(1, len(node_key_filter))