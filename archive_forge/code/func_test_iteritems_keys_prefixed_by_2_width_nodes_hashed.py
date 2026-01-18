from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iteritems_keys_prefixed_by_2_width_nodes_hashed(self):
    search_key_func = chk_map.search_key_registry.get(b'hash-16-way')
    self.assertEqual(b'E8B7BE43\x00E8B7BE43', search_key_func(StaticTuple(b'a', b'a')))
    self.assertEqual(b'E8B7BE43\x0071BEEFF9', search_key_func(StaticTuple(b'a', b'b')))
    self.assertEqual(b'71BEEFF9\x0000000000', search_key_func(StaticTuple(b'b', b'')))
    chkmap = self._get_map({(b'a', b'a'): b'content here', (b'a', b'b'): b'more content', (b'b', b''): b'boring content'}, maximum_size=10, key_width=2, search_key_func=search_key_func)
    self.assertEqual({(b'a', b'a'): b'content here', (b'a', b'b'): b'more content'}, self.to_dict(chkmap, [(b'a',)]))