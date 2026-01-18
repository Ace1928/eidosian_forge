import time
from . import debug, errors, osutils, revision, trace
def find_seen_ancestors(self, revisions):
    """Find ancestors of these revisions that have already been seen.

        This function generally makes the assumption that querying for the
        parents of a node that has already been queried is reasonably cheap.
        (eg, not a round trip to a remote host).
        """
    all_seen = self.seen
    pending = set(revisions).intersection(all_seen)
    seen_ancestors = set(pending)
    if self._returning == 'next':
        not_searched_yet = self._next_query
    else:
        not_searched_yet = ()
    pending.difference_update(not_searched_yet)
    get_parent_map = self._parents_provider.get_parent_map
    while pending:
        parent_map = get_parent_map(pending)
        all_parents = []
        for parent_ids in parent_map.values():
            all_parents.extend(parent_ids)
        next_pending = all_seen.intersection(all_parents).difference(seen_ancestors)
        seen_ancestors.update(next_pending)
        next_pending.difference_update(not_searched_yet)
        pending = next_pending
    return seen_ancestors