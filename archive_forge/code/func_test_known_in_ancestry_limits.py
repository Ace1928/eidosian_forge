from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_known_in_ancestry_limits(self):
    graph = self.make_breaking_graph(ancestry_1, [b'rev1'])
    self.assertFindDistance(4, graph, b'rev4', [(b'rev3', 3)])