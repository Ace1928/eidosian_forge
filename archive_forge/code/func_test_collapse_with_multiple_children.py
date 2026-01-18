from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_collapse_with_multiple_children(self):
    d = {1: [2, 3], 2: [4], 4: [6], 3: [5], 5: [6], 6: [7], 7: []}
    self.assertCollapsed(d, d)