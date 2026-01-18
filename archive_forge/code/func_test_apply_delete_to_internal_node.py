from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_apply_delete_to_internal_node(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(100)
    chkmap.map((b'small',), b'value')
    chkmap.map((b'little',), b'value')
    chkmap.map((b'very-big',), b'x' * 100)
    self.assertIsInstance(chkmap._root_node, InternalNode)
    delta = [((b'very-big',), None, None)]
    chkmap.apply_delta(delta)
    self.assertCanonicalForm(chkmap)
    self.assertIsInstance(chkmap._root_node, LeafNode)