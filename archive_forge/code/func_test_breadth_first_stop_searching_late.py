from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_stop_searching_late(self):
    graph = self.make_graph({b'head': [b'middle'], b'middle': [b'child'], b'child': [NULL_REVISION], NULL_REVISION: []})
    search = graph._make_breadth_first_searcher([b'head'])
    expected = [({b'head'}, ({b'head'}, {b'middle'}, 1), [b'head'], None, None), ({b'head', b'middle'}, ({b'head'}, {b'child'}, 2), [b'head', b'middle'], None, None), ({b'head', b'middle', b'child'}, ({b'head'}, {b'middle', b'child'}, 1), [b'head'], None, [b'middle', b'child'])]
    self.assertSeenAndResult(expected, search, search.__next__)
    search = graph._make_breadth_first_searcher([b'head'])
    self.assertSeenAndResult(expected, search, search.next_with_ghosts)