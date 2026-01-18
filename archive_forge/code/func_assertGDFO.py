import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def assertGDFO(self, graph, rev, gdfo):
    node = graph._nodes[rev]
    self.assertEqual(gdfo, node.gdfo)