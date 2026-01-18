from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def assertFindDistance(self, revno, graph, target_id, known_ids):
    """Assert the output of Graph.find_distance_to_null()"""
    actual = graph.find_distance_to_null(target_id, known_ids)
    self.assertEqual(revno, actual)