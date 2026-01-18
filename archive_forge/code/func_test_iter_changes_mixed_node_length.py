from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iter_changes_mixed_node_length(self):
    basis_dict = {(b'aaa',): b'foo bar', (b'aab',): b'common altered a', (b'b',): b'foo bar b'}
    target_dict = {(b'aaa',): b'foo bar', (b'aab',): b'common altered b', (b'at',): b'foo bar t'}
    changes = [((b'aab',), b'common altered a', b'common altered b'), ((b'at',), None, b'foo bar t'), ((b'b',), b'foo bar b', None)]
    basis = self._get_map(basis_dict, maximum_size=10)
    target = self._get_map(target_dict, maximum_size=10, chk_bytes=basis._store)
    self.assertEqual(changes, sorted(list(target.iter_changes(basis))))