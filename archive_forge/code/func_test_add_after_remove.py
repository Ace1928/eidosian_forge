from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_after_remove(self):
    tree = self.make_branch_and_tree('.')
    if not tree._format.supports_versioned_directories:
        raise tests.TestNotApplicable('format does not support versioned directories')
    self.build_tree(['dir/', 'dir/subdir/', 'dir/subdir/foo'])
    tree.add(['dir'])
    tree.commit('dir')
    tree.unversion(['dir'])
    self.assertRaises(errors.NotVersionedError, tree.add, ['dir/subdir/foo'])