from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def assertMergeOrder(self, expected, graph, tip, base_revisions):
    self.assertEqual(expected, graph.find_merge_order(tip, base_revisions))