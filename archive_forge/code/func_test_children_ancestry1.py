import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_children_ancestry1(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    self.assertEqual([b'rev1'], graph.get_child_keys(NULL_REVISION))
    self.assertEqual([b'rev2a', b'rev2b'], sorted(graph.get_child_keys(b'rev1')))
    self.assertEqual([b'rev3'], graph.get_child_keys(b'rev2a'))
    self.assertEqual([b'rev4'], graph.get_child_keys(b'rev3'))
    self.assertEqual([b'rev4'], graph.get_child_keys(b'rev2b'))
    self.assertRaises(KeyError, graph.get_child_keys, b'not_in_graph')