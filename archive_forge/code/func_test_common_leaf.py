from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_common_leaf(self):
    basis = self.get_map_key({})
    target1 = self.get_map_key({(b'aaa',): b'common'})
    target2 = self.get_map_key({(b'aaa',): b'common', (b'bbb',): b'new'})
    target3 = self.get_map_key({(b'aaa',): b'common', (b'aac',): b'other', (b'bbb',): b'new'})
    target1_map = CHKMap(self.get_chk_bytes(), target1)
    self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'common'\n", target1_map._dump_tree())
    target2_map = CHKMap(self.get_chk_bytes(), target2)
    self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'common'\n  'b' LeafNode\n      ('bbb',) 'new'\n", target2_map._dump_tree())
    target3_map = CHKMap(self.get_chk_bytes(), target3)
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aac' LeafNode\n      ('aac',) 'other'\n  'b' LeafNode\n      ('bbb',) 'new'\n", target3_map._dump_tree())
    aaa_key = target1_map._root_node.key()
    b_key = target2_map._root_node._items[b'b'].key()
    a_key = target3_map._root_node._items[b'a'].key()
    aac_key = target3_map._root_node._items[b'a']._items[b'aac'].key()
    self.assertIterInteresting([target1, target2, target3, a_key, aac_key, b_key], [((b'aaa',), b'common'), ((b'bbb',), b'new'), ((b'aac',), b'other')], [target1, target2, target3], [basis])
    self.assertIterInteresting([target2, target3, a_key, aac_key, b_key], [((b'bbb',), b'new'), ((b'aac',), b'other')], [target2, target3], [target1])
    self.assertIterInteresting([target1], [], [target1], [target3])