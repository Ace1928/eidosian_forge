from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iteritems_two_items(self):
    chk_bytes = self.get_chk_bytes()
    root_key = CHKMap.from_dict(chk_bytes, {(b'a',): b'content here', (b'b',): b'more content'})
    chkmap = CHKMap(chk_bytes, root_key)
    self.assertEqual([((b'a',), b'content here'), ((b'b',), b'more content')], sorted(list(chkmap.iteritems())))