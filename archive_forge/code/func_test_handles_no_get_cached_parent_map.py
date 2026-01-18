from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_handles_no_get_cached_parent_map(self):
    pp1 = self.get_shared_provider(b'pp1', {b'rev2': (b'rev1',)}, has_cached=False)
    pp2 = self.get_shared_provider(b'pp2', {b'rev2': (b'rev1',)}, has_cached=True)
    stacked = _mod_graph.StackedParentsProvider([pp1, pp2])
    self.assertEqual({b'rev2': (b'rev1',)}, stacked.get_parent_map([b'rev2']))
    self.assertEqual([(b'pp2', 'cached', [b'rev2'])], self.calls)