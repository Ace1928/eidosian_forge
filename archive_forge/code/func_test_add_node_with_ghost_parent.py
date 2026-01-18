import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_add_node_with_ghost_parent(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    graph.add_node(b'rev5', [b'rev2b', b'revGhost'])
    self.assertGDFO(graph, b'rev5', 4)
    self.assertGDFO(graph, b'revGhost', 1)