from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iteritems_partial_empty(self):
    node = InternalNode()
    self.assertEqual([], sorted(node.iteritems([(b'missing',)])))