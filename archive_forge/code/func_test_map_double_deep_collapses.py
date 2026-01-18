from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_double_deep_collapses(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(40)
    chkmap.map((b'aaa',), b'v')
    chkmap.map((b'aab',), b'very long value that splits')
    chkmap.map((b'abc',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aa' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n    'aab' LeafNode\n      ('aab',) 'very long value that splits'\n  'ab' LeafNode\n      ('abc',) 'v'\n", chkmap._dump_tree())
    chkmap.map((b'aab',), b'v')
    self.assertCanonicalForm(chkmap)
    self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aab',) 'v'\n      ('abc',) 'v'\n", chkmap._dump_tree())