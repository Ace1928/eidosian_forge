from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_deserialise_item_with_common_prefix(self):
    node = self.module._deserialise_leaf_node(b'chkleaf:\n0\n2\n2\nfoo\x00\n1\x001\nbar\x00baz\n2\x001\nblarh\n', (b'sha1:1234',))
    self.assertEqual(2, len(node))
    self.assertEqual([((b'foo', b'1'), b'bar\x00baz'), ((b'foo', b'2'), b'blarh')], sorted(node.iteritems(None)))
    self.assertIs(chk_map._unknown, node._search_prefix)
    self.assertEqual(b'foo\x00', node._common_serialised_prefix)