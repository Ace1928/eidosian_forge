from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_deserialise_item_with_null_width_1(self):
    node = self.module._deserialise_leaf_node(b'chkleaf:\n0\n1\n2\n\nfoo\x001\nbar\x00baz\nquux\x001\nblarh\n', (b'sha1:1234',))
    self.assertEqual(2, len(node))
    self.assertEqual([((b'foo',), b'bar\x00baz'), ((b'quux',), b'blarh')], sorted(node.iteritems(None)))