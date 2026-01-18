from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
def get_rootfull_delta(self, repository, revid):
    tree = repository.revision_tree(revid)
    with repository.lock_read():
        parent_revid = repository.get_parent_map([revid])[revid][0]
        basis_tree = repository.revision_tree(parent_revid)
        tree = repository.revision_tree(revid)
        return tree.changes_from(basis_tree, include_root=True)