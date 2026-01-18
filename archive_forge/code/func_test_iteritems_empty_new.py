from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iteritems_empty_new(self):
    node = InternalNode()
    self.assertEqual([], sorted(node.iteritems(None)))