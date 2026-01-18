from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_ancestor_is_ghost(self):
    graph = self.make_graph({b'rev': [b'parent']})
    e = self.assertRaises(errors.GhostRevisionsHaveNoRevno, graph.find_distance_to_null, b'rev', [])
    self.assertEqual(b'rev', e.revision_id)
    self.assertEqual(b'parent', e.ghost_revision_id)