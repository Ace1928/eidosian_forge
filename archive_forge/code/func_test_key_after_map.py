from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_key_after_map(self):
    node = self.module._deserialise_leaf_node(b'chkleaf:\n10\n1\n0\n\n', (b'sha1:1234',))
    node.map(None, (b'foo bar',), b'baz quux')
    self.assertEqual(None, node.key())