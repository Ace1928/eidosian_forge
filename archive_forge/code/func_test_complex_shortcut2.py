from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_complex_shortcut2(self):
    graph = self.make_graph(complex_shortcut2)
    self.assertFindUniqueAncestors(graph, [b'j', b'u'], b'u', [b't'])
    self.assertFindUniqueAncestors(graph, [b't'], b't', [b'u'])