import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_add_with_all_ghost_parents(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    graph.add_node(b'rev5', [b'ghost'])
    self.assertGDFO(graph, b'rev5', 2)
    self.assertGDFO(graph, b'ghost', 1)