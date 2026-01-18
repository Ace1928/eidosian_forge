from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_stop_searching_not_queried(self):
    graph = self.make_graph({b'head': [b'child', b'ghost1'], b'child': [NULL_REVISION], NULL_REVISION: []})
    search = graph._make_breadth_first_searcher([b'head'])
    expected = [({b'head'}, ({b'head'}, {b'child', NULL_REVISION, b'ghost1'}, 1), [b'head'], None, [NULL_REVISION, b'ghost1']), ({b'head', b'child', b'ghost1'}, ({b'head'}, {b'ghost1', NULL_REVISION}, 2), [b'head', b'child'], None, [NULL_REVISION, b'ghost1'])]
    self.assertSeenAndResult(expected, search, search.__next__)
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertSeenAndResult(expected, search, search.next_with_ghosts)