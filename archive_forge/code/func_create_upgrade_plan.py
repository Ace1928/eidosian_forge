from ... import osutils, trace, ui
from ...errors import BzrError
from .rebase import (CommitBuilderRevisionRewriter, generate_transpose_plan,
def create_upgrade_plan(repository, generate_rebase_map, determine_new_revid, revision_id=None, allow_changes=False):
    """Generate a rebase plan for upgrading revisions.

    :param repository: Repository to do upgrade in
    :param foreign_repository: Subversion repository to fetch new revisions
        from.
    :param new_mapping: New mapping to use.
    :param revision_id: Revision to upgrade (None for all revisions in
        repository.)
    :param allow_changes: Whether an upgrade is allowed to change the contents
        of revisions.
    :return: Tuple with a rebase plan and map of renamed revisions.
    """
    graph = repository.get_graph()
    upgrade_map = generate_rebase_map(revision_id)
    if not allow_changes:
        for oldrevid, newrevid in upgrade_map.iteritems():
            oldrev = repository.get_revision(oldrevid)
            newrev = repository.get_revision(newrevid)
            check_revision_changed(oldrev, newrev)
    if revision_id is None:
        heads = repository.all_revision_ids()
    else:
        heads = [revision_id]
    plan = generate_transpose_plan(graph.iter_ancestry(heads), upgrade_map, graph, determine_new_revid)

    def remove_parents(entry):
        oldrevid, (newrevid, parents) = entry
        return (oldrevid, newrevid)
    upgrade_map.update(dict(map(remove_parents, plan.iteritems())))
    return (plan, upgrade_map)