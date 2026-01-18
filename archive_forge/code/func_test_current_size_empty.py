from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_current_size_empty(self):
    node = LeafNode()
    self.assertEqual(16, node._current_size())