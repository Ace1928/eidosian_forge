from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_apply_new_keys_must_be_new(self):
    chk_bytes = self.get_chk_bytes()
    root_key = CHKMap.from_dict(chk_bytes, {(b'a',): b'b'})
    chkmap = CHKMap(chk_bytes, root_key)
    self.assertRaises(errors.InconsistentDelta, chkmap.apply_delta, [(None, (b'a',), b'b')])
    self.assertEqual(root_key, chkmap._root_node._key)