from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test__check_with_refs(self):
    tree = self.make_branch_and_tree('tree')
    if not isinstance(tree, InventoryWorkingTree):
        raise TestNotApplicable('_get_check_refs only relevant for inventory working trees')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    revid = tree.commit('first post')
    needed_refs = tree._get_check_refs()
    repo = tree.branch.repository
    for ref in needed_refs:
        kind, revid = ref
        refs = {}
        if kind == 'trees':
            refs[ref] = repo.revision_tree(revid)
        else:
            self.fail('unknown ref kind')
    tree._check(refs)