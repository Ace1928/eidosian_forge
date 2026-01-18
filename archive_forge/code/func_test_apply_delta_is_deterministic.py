from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_apply_delta_is_deterministic(self):
    chk_bytes = self.get_chk_bytes()
    chkmap1 = CHKMap(chk_bytes, None)
    chkmap1._root_node.set_maximum_size(10)
    chkmap1.apply_delta([(None, (b'aaa',), b'common'), (None, (b'bba',), b'target2'), (None, (b'bbb',), b'common')])
    root_key1 = chkmap1._save()
    self.assertCanonicalForm(chkmap1)
    chkmap2 = CHKMap(chk_bytes, None)
    chkmap2._root_node.set_maximum_size(10)
    chkmap2.apply_delta([(None, (b'bbb',), b'common'), (None, (b'bba',), b'target2'), (None, (b'aaa',), b'common')])
    root_key2 = chkmap2._save()
    self.assertEqualDiff(chkmap1._dump_tree(include_keys=True), chkmap2._dump_tree(include_keys=True))
    self.assertEqual(root_key1, root_key2)
    self.assertCanonicalForm(chkmap2)