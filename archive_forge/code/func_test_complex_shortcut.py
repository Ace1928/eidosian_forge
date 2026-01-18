from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_complex_shortcut(self):
    graph = self.make_graph(complex_shortcut)
    self.assertFindUniqueAncestors(graph, [b'h', b'n'], b'n', [b'm'])
    self.assertFindUniqueAncestors(graph, [b'e', b'i', b'm'], b'm', [b'n'])