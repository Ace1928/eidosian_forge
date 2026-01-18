from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_init_and_save_new(self):
    chk_bytes = self.get_chk_bytes()
    chkmap = CHKMap(chk_bytes, None)
    key = chkmap._save()
    leaf_node = LeafNode()
    self.assertEqual([key], leaf_node.serialise(chk_bytes))