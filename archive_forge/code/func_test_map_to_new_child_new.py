from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_to_new_child_new(self):
    chkmap = self._get_map({(b'k1',): b'foo', (b'k2',): b'bar'}, maximum_size=10)
    chkmap._ensure_root()
    node = chkmap._root_node
    self.assertEqual(2, len([value for value in node._items.values() if isinstance(value, StaticTuple)]))
    prefix, nodes = node.map(None, (b'k3',), b'quux')
    self.assertEqual(b'k', prefix)
    self.assertEqual([(b'', node)], nodes)
    child = node._items[b'k3']
    self.assertIsInstance(child, LeafNode)
    self.assertEqual(1, len(child))
    self.assertEqual({(b'k3',): b'quux'}, self.to_dict(child, None))
    self.assertEqual(None, child._key)
    self.assertEqual(10, child.maximum_size)
    self.assertEqual(1, child._key_width)
    self.assertEqual(3, len(chkmap))
    self.assertEqual({(b'k1',): b'foo', (b'k2',): b'bar', (b'k3',): b'quux'}, self.to_dict(chkmap))
    keys = list(node.serialise(chkmap._store))
    child_key = child.serialise(chkmap._store)[0]
    self.assertEqual([child_key, keys[1]], keys)