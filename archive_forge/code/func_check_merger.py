from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def check_merger(self, result, ancestry, merged, tip):
    graph = self.make_graph(ancestry)
    self.assertEqual(result, graph.find_lefthand_merger(merged, tip))