from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_multiple_maps_similar(self):
    basis = self.get_map_key({(b'aaa',): b'unchanged', (b'abb',): b'will change left', (b'caa',): b'unchanged', (b'cbb',): b'will change right'}, maximum_size=60)
    left = self.get_map_key({(b'aaa',): b'unchanged', (b'abb',): b'changed left', (b'caa',): b'unchanged', (b'cbb',): b'will change right'}, maximum_size=60)
    right = self.get_map_key({(b'aaa',): b'unchanged', (b'abb',): b'will change left', (b'caa',): b'unchanged', (b'cbb',): b'changed right'}, maximum_size=60)
    basis_map = CHKMap(self.get_chk_bytes(), basis)
    self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'will change left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'will change right'\n", basis_map._dump_tree())
    left_map = CHKMap(self.get_chk_bytes(), left)
    self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'changed left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'will change right'\n", left_map._dump_tree())
    l_a_key = left_map._root_node._items[b'a'].key()
    l_c_key = left_map._root_node._items[b'c'].key()
    right_map = CHKMap(self.get_chk_bytes(), right)
    self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'unchanged'\n      ('abb',) 'will change left'\n  'c' LeafNode\n      ('caa',) 'unchanged'\n      ('cbb',) 'changed right'\n", right_map._dump_tree())
    r_a_key = right_map._root_node._items[b'a'].key()
    r_c_key = right_map._root_node._items[b'c'].key()
    self.assertIterInteresting([right, left, l_a_key, r_c_key], [((b'abb',), b'changed left'), ((b'cbb',), b'changed right')], [left, right], [basis])