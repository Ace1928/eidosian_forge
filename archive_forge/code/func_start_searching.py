import time
from . import debug, errors, osutils, revision, trace
def start_searching(self, revisions):
    """Add revisions to the search.

        The parents of revisions will be returned from the next call to next()
        or next_with_ghosts(). If next_with_ghosts was the most recently used
        next* call then the return value is the result of looking up the
        ghost/not ghost status of revisions. (A tuple (present, ghosted)).
        """
    revisions = frozenset(revisions)
    self._started_keys.update(revisions)
    new_revisions = revisions.difference(self.seen)
    if self._returning == 'next':
        self._next_query.update(new_revisions)
        self.seen.update(new_revisions)
    else:
        revs, ghosts, query, parents = self._do_query(revisions)
        self._stopped_keys.update(ghosts)
        self._current_present.update(revs)
        self._current_ghosts.update(ghosts)
        self._next_query.update(query)
        self._current_parents.update(parents)
        return (revs, ghosts)