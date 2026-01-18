import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_heads_criss_cross(self):
    graph = self.make_known_graph(test_graph.criss_cross)
    self.assertEqual({b'rev2a'}, graph.heads([b'rev2a', b'rev1']))
    self.assertEqual({b'rev2b'}, graph.heads([b'rev2b', b'rev1']))
    self.assertEqual({b'rev3a'}, graph.heads([b'rev3a', b'rev1']))
    self.assertEqual({b'rev3b'}, graph.heads([b'rev3b', b'rev1']))
    self.assertEqual({b'rev2a', b'rev2b'}, graph.heads([b'rev2a', b'rev2b']))
    self.assertEqual({b'rev3a'}, graph.heads([b'rev3a', b'rev2a']))
    self.assertEqual({b'rev3a'}, graph.heads([b'rev3a', b'rev2b']))
    self.assertEqual({b'rev3a'}, graph.heads([b'rev3a', b'rev2a', b'rev2b']))
    self.assertEqual({b'rev3b'}, graph.heads([b'rev3b', b'rev2a']))
    self.assertEqual({b'rev3b'}, graph.heads([b'rev3b', b'rev2b']))
    self.assertEqual({b'rev3b'}, graph.heads([b'rev3b', b'rev2a', b'rev2b']))
    self.assertEqual({b'rev3a', b'rev3b'}, graph.heads([b'rev3a', b'rev3b']))
    self.assertEqual({b'rev3a', b'rev3b'}, graph.heads([b'rev3a', b'rev3b', b'rev2a', b'rev2b']))