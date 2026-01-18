from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_get_result_excludes_current_pending(self):
    graph = self.make_graph({b'head': [b'child'], b'child': [NULL_REVISION], NULL_REVISION: []})
    search = graph._make_breadth_first_searcher([b'head'])
    state = search.get_state()
    self.assertEqual(({b'head'}, {b'head'}, set()), state)
    self.assertEqual(set(), search.seen)
    expected = [({b'head'}, ({b'head'}, {b'child'}, 1), [b'head'], None, None), ({b'head', b'child'}, ({b'head'}, {NULL_REVISION}, 2), [b'head', b'child'], None, None), ({b'head', b'child', NULL_REVISION}, ({b'head'}, set(), 3), [b'head', b'child', NULL_REVISION], None, None)]
    self.assertSeenAndResult(expected, search, search.__next__)
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertSeenAndResult(expected, search, search.next_with_ghosts)