import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_add_existing_node_mismatched_parents(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    self.assertRaises(ValueError, graph.add_node, b'rev4', [b'rev2b', b'rev3'])