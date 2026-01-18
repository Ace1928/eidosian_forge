import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_filling_in_ghosts_resets_head_cache(self):
    graph = self.make_known_graph(test_graph.with_ghost)
    self.assertEqual({b'e', b'g'}, graph.heads([b'e', b'g']))
    graph.add_node(b'g', [b'e'])
    self.assertEqual({b'g'}, graph.heads([b'e', b'g']))