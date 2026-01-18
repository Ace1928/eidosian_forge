from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_graph_difference_double_shortcut(self):
    graph = self.make_graph(double_shortcut)
    self.assertEqual(({b'd', b'f'}, {b'e', b'g'}), graph.find_difference(b'f', b'g'))