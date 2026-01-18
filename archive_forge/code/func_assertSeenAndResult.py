from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def assertSeenAndResult(self, instructions, search, next):
    """Check the results of .seen and get_result() for a seach.

        :param instructions: A list of tuples:
            (seen, recipe, included_keys, starts, stops).
            seen, recipe and included_keys are results to check on the search
            and the searches get_result(). starts and stops are parameters to
            pass to start_searching and stop_searching_any during each
            iteration, if they are not None.
        :param search: The search to use.
        :param next: A callable to advance the search.
        """
    for seen, recipe, included_keys, starts, stops in instructions:
        recipe = ('search',) + recipe
        next()
        if starts is not None:
            search.start_searching(starts)
        if stops is not None:
            search.stop_searching_any(stops)
        state = search.get_state()
        self.assertEqual(set(included_keys), state[2])
        self.assertEqual(seen, search.seen)