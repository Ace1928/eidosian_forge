import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_gdfo_after_add_node(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    self.assertEqual([], graph.get_child_keys(b'rev4'))
    graph.add_node(b'rev5', [b'rev4'])
    self.assertEqual([b'rev4'], graph.get_parent_keys(b'rev5'))
    self.assertEqual([b'rev5'], graph.get_child_keys(b'rev4'))
    self.assertEqual([], graph.get_child_keys(b'rev5'))
    self.assertGDFO(graph, b'rev5', 6)
    graph.add_node(b'rev6', [b'rev2b'])
    graph.add_node(b'rev7', [b'rev6'])
    graph.add_node(b'rev8', [b'rev7', b'rev5'])
    self.assertGDFO(graph, b'rev5', 6)
    self.assertGDFO(graph, b'rev6', 4)
    self.assertGDFO(graph, b'rev7', 5)
    self.assertGDFO(graph, b'rev8', 7)