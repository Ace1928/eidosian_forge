import time
from . import debug, errors, osutils, revision, trace
def find_unique_ancestors(self, unique_revision, common_revisions):
    """Find the unique ancestors for a revision versus others.

        This returns the ancestry of unique_revision, excluding all revisions
        in the ancestry of common_revisions. If unique_revision is in the
        ancestry, then the empty set will be returned.

        :param unique_revision: The revision_id whose ancestry we are
            interested in.
            (XXX: Would this API be better if we allowed multiple revisions on
            to be searched here?)
        :param common_revisions: Revision_ids of ancestries to exclude.
        :return: A set of revisions in the ancestry of unique_revision
        """
    if unique_revision in common_revisions:
        return set()
    unique_searcher, common_searcher = self._find_initial_unique_nodes([unique_revision], common_revisions)
    unique_nodes = unique_searcher.seen.difference(common_searcher.seen)
    if not unique_nodes:
        return unique_nodes
    all_unique_searcher, unique_tip_searchers = self._make_unique_searchers(unique_nodes, unique_searcher, common_searcher)
    self._refine_unique_nodes(unique_searcher, all_unique_searcher, unique_tip_searchers, common_searcher)
    true_unique_nodes = unique_nodes.difference(common_searcher.seen)
    if 'graph' in debug.debug_flags:
        trace.mutter('Found %d truly unique nodes out of %d', len(true_unique_nodes), len(unique_nodes))
    return true_unique_nodes