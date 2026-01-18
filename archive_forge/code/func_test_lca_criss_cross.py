from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_lca_criss_cross(self):
    """Test least-common-ancestor after a criss-cross merge."""
    graph = self.make_graph(criss_cross)
    self.assertEqual({b'rev2a', b'rev2b'}, graph.find_lca(b'rev3a', b'rev3b'))
    self.assertEqual({b'rev2b'}, graph.find_lca(b'rev3a', b'rev3b', b'rev2b'))