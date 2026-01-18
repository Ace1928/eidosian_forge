from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_first_item_new(self):
    chk_bytes = self.get_chk_bytes()
    chkmap = CHKMap(chk_bytes, None)
    chkmap.map((b'foo,',), b'bar')
    self.assertEqual({(b'foo,',): b'bar'}, self.to_dict(chkmap))
    self.assertEqual(1, len(chkmap))
    key = chkmap._save()
    leaf_node = LeafNode()
    leaf_node.map(chk_bytes, (b'foo,',), b'bar')
    self.assertEqual([key], leaf_node.serialise(chk_bytes))