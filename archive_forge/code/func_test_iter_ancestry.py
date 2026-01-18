from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_iter_ancestry(self):
    nodes = boundary.copy()
    nodes[NULL_REVISION] = ()
    graph = self.make_graph(nodes)
    expected = nodes.copy()
    expected.pop(b'a')
    self.assertEqual(expected, dict(graph.iter_ancestry([b'c'])))
    self.assertEqual(nodes, dict(graph.iter_ancestry([b'a', b'c'])))