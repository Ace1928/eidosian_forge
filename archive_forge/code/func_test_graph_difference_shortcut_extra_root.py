from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_graph_difference_shortcut_extra_root(self):
    graph = self.make_graph(shortcut_extra_root)
    self.assertEqual(({b'e'}, {b'f', b'g'}), graph.find_difference(b'e', b'f'))