from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_from_dict_empty(self):
    chk_bytes = self.get_chk_bytes()
    root_key = CHKMap.from_dict(chk_bytes, {})
    expected_root_key = self.assertHasEmptyMap(chk_bytes)
    self.assertEqual(expected_root_key, root_key)