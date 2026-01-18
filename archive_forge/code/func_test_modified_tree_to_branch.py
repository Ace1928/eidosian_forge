from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_modified_tree_to_branch(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    tree.add('file')
    reconfiguration = reconfigure.Reconfigure.to_branch(tree.controldir)
    self.assertRaises(errors.UncommittedChanges, reconfiguration.apply)
    reconfiguration.apply(force=True)
    self.assertRaises(errors.NoWorkingTree, workingtree.WorkingTree.open, 'tree')