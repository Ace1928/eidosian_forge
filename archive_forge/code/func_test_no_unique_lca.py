from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_no_unique_lca(self):
    """Test error when one revision is not in the graph"""
    graph = self.make_graph(ancestry_1)
    self.assertRaises(errors.NoCommonAncestor, graph.find_unique_lca, b'rev1', b'1rev')