import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_gdfo_ancestry_1(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    self.assertGDFO(graph, b'rev1', 2)
    self.assertGDFO(graph, b'rev2b', 3)
    self.assertGDFO(graph, b'rev2a', 3)
    self.assertGDFO(graph, b'rev3', 4)
    self.assertGDFO(graph, b'rev4', 5)