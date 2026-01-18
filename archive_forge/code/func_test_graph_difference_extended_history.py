from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_graph_difference_extended_history(self):
    graph = self.make_graph(extended_history_shortcut)
    self.assertEqual(({b'e'}, {b'f'}), graph.find_difference(b'e', b'f'))
    self.assertEqual(({b'f'}, {b'e'}), graph.find_difference(b'f', b'e'))