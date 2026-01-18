import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_gdfo_extended_history_shortcut(self):
    graph = self.make_known_graph(test_graph.extended_history_shortcut)
    self.assertGDFO(graph, b'a', 2)
    self.assertGDFO(graph, b'b', 3)
    self.assertGDFO(graph, b'c', 4)
    self.assertGDFO(graph, b'd', 5)
    self.assertGDFO(graph, b'e', 6)
    self.assertGDFO(graph, b'f', 6)