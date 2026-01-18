from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_graph_difference_separate_ancestry(self):
    graph = self.make_graph(ancestry_2)
    self.assertEqual(({b'rev1a'}, {b'rev1b'}), graph.find_difference(b'rev1a', b'rev1b'))
    self.assertEqual(({b'rev1a', b'rev2a', b'rev3a', b'rev4a'}, {b'rev1b'}), graph.find_difference(b'rev4a', b'rev1b'))