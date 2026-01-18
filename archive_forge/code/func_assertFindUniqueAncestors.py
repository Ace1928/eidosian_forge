from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def assertFindUniqueAncestors(self, graph, expected, node, common):
    actual = graph.find_unique_ancestors(node, common)
    self.assertEqual(expected, sorted(actual))