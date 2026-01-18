from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_multiple_maps(self):
    basis1 = self.get_map_key({(b'aaa',): b'common', (b'aab',): b'basis1'})
    basis2 = self.get_map_key({(b'bbb',): b'common', (b'bbc',): b'basis2'})
    target1 = self.get_map_key({(b'aaa',): b'common', (b'aac',): b'target1', (b'bbb',): b'common'})
    target2 = self.get_map_key({(b'aaa',): b'common', (b'bba',): b'target2', (b'bbb',): b'common'})
    target1_map = CHKMap(self.get_chk_bytes(), target1)
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'common'\n    'aac' LeafNode\n      ('aac',) 'target1'\n  'b' LeafNode\n      ('bbb',) 'common'\n", target1_map._dump_tree())
    a_key = target1_map._root_node._items[b'a'].key()
    aac_key = target1_map._root_node._items[b'a']._items[b'aac'].key()
    target2_map = CHKMap(self.get_chk_bytes(), target2)
    self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'common'\n  'b' InternalNode\n    'bba' LeafNode\n      ('bba',) 'target2'\n    'bbb' LeafNode\n      ('bbb',) 'common'\n", target2_map._dump_tree())
    b_key = target2_map._root_node._items[b'b'].key()
    bba_key = target2_map._root_node._items[b'b']._items[b'bba'].key()
    self.assertIterInteresting([target1, target2, a_key, aac_key, b_key, bba_key], [((b'aac',), b'target1'), ((b'bba',), b'target2')], [target1, target2], [basis1, basis2])