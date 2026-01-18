from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_lefthand_distance_smoke(self):
    """A simple does it work test for graph.lefthand_distance(keys)."""
    graph = self.make_graph(history_shortcut)
    distance_graph = graph.find_lefthand_distances([b'rev3b', b'rev2a'])
    self.assertEqual({b'rev2a': 2, b'rev3b': 3}, distance_graph)