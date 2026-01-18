import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_heads_alt_merge(self):
    graph = self.make_known_graph(alt_merge)
    self.assertEqual({b'c'}, graph.heads([b'a', b'c']))