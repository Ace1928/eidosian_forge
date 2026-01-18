from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iteritems_two_children_with_hash(self):
    search_key_func = chk_map.search_key_registry.get(b'hash-255-way')
    node = InternalNode(search_key_func=search_key_func)
    leaf1 = LeafNode(search_key_func=search_key_func)
    leaf1.map(None, StaticTuple(b'foo bar'), b'quux')
    leaf2 = LeafNode(search_key_func=search_key_func)
    leaf2.map(None, StaticTuple(b'strange'), b'beast')
    self.assertEqual(b'\xbeF\x014', search_key_func(StaticTuple(b'foo bar')))
    self.assertEqual(b'\x85\xfa\xf7K', search_key_func(StaticTuple(b'strange')))
    node.add_node(b'\xbe', leaf1)
    node._items[b'\xbe'] = None
    node.add_node(b'\x85', leaf2)
    self.assertEqual([((b'strange',), b'beast')], sorted(node.iteritems(None, [StaticTuple(b'strange'), StaticTuple(b'weird')])))