from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_max_size_100_bytes_new(self):
    chkmap = self._get_map({(b'k1' * 50,): b'v1', (b'k2' * 50,): b'v2'}, maximum_size=100)
    chkmap._ensure_root()
    self.assertEqual(100, chkmap._root_node.maximum_size)
    self.assertEqual(1, chkmap._root_node._key_width)
    self.assertEqual(2, len(chkmap._root_node._items))
    self.assertEqual(b'k', chkmap._root_node._compute_search_prefix())
    nodes = sorted(chkmap._root_node._items.items())
    ptr1 = nodes[0]
    ptr2 = nodes[1]
    self.assertEqual(b'k1', ptr1[0])
    self.assertEqual(b'k2', ptr2[0])
    node1 = chk_map._deserialise(chkmap._read_bytes(ptr1[1]), ptr1[1], None)
    self.assertIsInstance(node1, LeafNode)
    self.assertEqual(1, len(node1))
    self.assertEqual({(b'k1' * 50,): b'v1'}, self.to_dict(node1, chkmap._store))
    node2 = chk_map._deserialise(chkmap._read_bytes(ptr2[1]), ptr2[1], None)
    self.assertIsInstance(node2, LeafNode)
    self.assertEqual(1, len(node2))
    self.assertEqual({(b'k2' * 50,): b'v2'}, self.to_dict(node2, chkmap._store))
    self.assertEqual(2, len(chkmap))
    self.assertEqual({(b'k1' * 50,): b'v1', (b'k2' * 50,): b'v2'}, self.to_dict(chkmap))