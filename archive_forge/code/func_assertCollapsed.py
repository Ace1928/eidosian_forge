from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def assertCollapsed(self, collapsed, original):
    self.assertEqual(collapsed, _mod_graph.collapse_linear_regions(original))