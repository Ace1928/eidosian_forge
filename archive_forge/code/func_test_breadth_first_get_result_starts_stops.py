from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_get_result_starts_stops(self):
    graph = self.make_graph({b'head': [b'child'], b'child': [NULL_REVISION], b'otherhead': [b'otherchild'], b'otherchild': [b'excluded'], b'excluded': [NULL_REVISION], NULL_REVISION: []})
    search = graph._make_breadth_first_searcher([])
    search.start_searching([b'head'])
    state = search.get_state()
    self.assertEqual(({b'head'}, {b'child'}, {b'head'}), state)
    self.assertEqual({b'head'}, search.seen)
    expected = [({b'head', b'child', b'otherhead'}, ({b'head', b'otherhead'}, {b'child', b'otherchild'}, 2), [b'head', b'otherhead'], [b'otherhead'], [b'child']), ({b'head', b'child', b'otherhead', b'otherchild'}, ({b'head', b'otherhead'}, {b'child', b'excluded'}, 3), [b'head', b'otherhead', b'otherchild'], None, None), ({b'head', b'child', b'otherhead', b'otherchild', b'excluded'}, ({b'head', b'otherhead'}, {b'child', b'excluded'}, 3), [b'head', b'otherhead', b'otherchild'], None, [b'excluded'])]
    self.assertSeenAndResult(expected, search, search.__next__)
    search = graph._make_breadth_first_searcher([])
    search.start_searching([b'head'])
    self.assertSeenAndResult(expected, search, search.next_with_ghosts)