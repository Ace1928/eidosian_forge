from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_find_descendants_rev1_rev4(self):
    graph = self.make_graph(ancestry_1)
    descendants = graph.find_descendants(b'rev1', b'rev4')
    self.assertEqual({b'rev1', b'rev2a', b'rev2b', b'rev3', b'rev4'}, descendants)