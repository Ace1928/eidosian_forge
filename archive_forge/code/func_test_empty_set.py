from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_empty_set(self):
    graph = self.make_graph(ancestry_1)
    self.assertFindUniqueAncestors(graph, [], b'rev1', [b'rev1'])
    self.assertFindUniqueAncestors(graph, [], b'rev2b', [b'rev2b'])
    self.assertFindUniqueAncestors(graph, [], b'rev3', [b'rev1', b'rev3'])