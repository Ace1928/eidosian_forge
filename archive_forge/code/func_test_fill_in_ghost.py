import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_fill_in_ghost(self):
    graph = self.make_known_graph(test_graph.with_ghost)
    graph.add_node(b'x', [])
    graph.add_node(b'y', [b'x'])
    graph.add_node(b'z', [b'y'])
    graph.add_node(b'g', [b'z'])
    self.assertGDFO(graph, b'f', 2)
    self.assertGDFO(graph, b'e', 3)
    self.assertGDFO(graph, b'x', 1)
    self.assertGDFO(graph, b'y', 2)
    self.assertGDFO(graph, b'z', 3)
    self.assertGDFO(graph, b'g', 4)
    self.assertGDFO(graph, b'b', 4)
    self.assertGDFO(graph, b'd', 5)
    self.assertGDFO(graph, b'a', 5)
    self.assertGDFO(graph, b'c', 6)