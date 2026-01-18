from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def make_fo_fa_node(self):
    node = InternalNode(b'f')
    child = LeafNode()
    child.set_maximum_size(100)
    child.map(None, (b'foo',), b'val')
    child.map(None, (b'fob',), b'val')
    node.add_node(b'fo', child)
    child = LeafNode()
    child.set_maximum_size(100)
    child.map(None, (b'far',), b'val')
    child.map(None, (b'faz',), b'val')
    node.add_node(b'fa', child)
    return node