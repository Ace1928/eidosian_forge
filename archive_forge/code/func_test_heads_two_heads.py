import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_heads_two_heads(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    self.assertEqual({b'rev2a', b'rev2b'}, graph.heads([b'rev2a', b'rev2b']))
    self.assertEqual({b'rev3', b'rev2b'}, graph.heads([b'rev3', b'rev2b']))