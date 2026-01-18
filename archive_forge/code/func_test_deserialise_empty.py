from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_deserialise_empty(self):
    node = self.module._deserialise_leaf_node(b'chkleaf:\n10\n1\n0\n\n', stuple(b'sha1:1234'))
    self.assertEqual(0, len(node))
    self.assertEqual(10, node.maximum_size)
    self.assertEqual((b'sha1:1234',), node.key())
    self.assertIsInstance(node.key(), StaticTuple)
    self.assertIs(None, node._search_prefix)
    self.assertIs(None, node._common_serialised_prefix)