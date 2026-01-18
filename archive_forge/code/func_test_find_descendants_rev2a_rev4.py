from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_find_descendants_rev2a_rev4(self):
    graph = self.make_graph(ancestry_1)
    descendants = graph.find_descendants(b'rev2a', b'rev4')
    self.assertEqual({b'rev2a', b'rev3', b'rev4'}, descendants)