from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_breadth_first_search_start_ghosts(self):
    graph = self.make_graph({})
    search = graph._make_breadth_first_searcher([b'a-ghost'])
    self.assertEqual((set(), {b'a-ghost'}), search.next_with_ghosts())
    self.assertRaises(StopIteration, search.next_with_ghosts)
    search = graph._make_breadth_first_searcher([b'a-ghost'])
    self.assertEqual({b'a-ghost'}, next(search))
    self.assertRaises(StopIteration, next, search)