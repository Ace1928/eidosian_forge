import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_heads_shortcut(self):
    graph = self.make_known_graph(test_graph.history_shortcut)
    self.assertEqual({b'rev2a', b'rev2b', b'rev2c'}, graph.heads([b'rev2a', b'rev2b', b'rev2c']))
    self.assertEqual({b'rev3a', b'rev3b'}, graph.heads([b'rev3a', b'rev3b']))
    self.assertEqual({b'rev3a', b'rev3b'}, graph.heads([b'rev2a', b'rev3a', b'rev3b']))
    self.assertEqual({b'rev2a', b'rev3b'}, graph.heads([b'rev2a', b'rev3b']))
    self.assertEqual({b'rev2c', b'rev3a'}, graph.heads([b'rev2c', b'rev3a']))