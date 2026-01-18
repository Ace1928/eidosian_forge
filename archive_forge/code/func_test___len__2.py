from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test___len__2(self):
    chkmap = self._get_map({(b'foo',): b'bar', (b'gam',): b'quux'})
    self.assertEqual(2, len(chkmap))