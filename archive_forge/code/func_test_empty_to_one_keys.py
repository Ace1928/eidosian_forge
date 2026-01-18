from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_empty_to_one_keys(self):
    target = self.get_map_key({(b'a',): b'content'})
    self.assertIterInteresting([target], [((b'a',), b'content')], [target], [])