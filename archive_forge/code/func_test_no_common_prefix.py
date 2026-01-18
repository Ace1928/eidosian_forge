from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_no_common_prefix(self):
    self.assertCommonPrefix(b'', b'begin', b'end')