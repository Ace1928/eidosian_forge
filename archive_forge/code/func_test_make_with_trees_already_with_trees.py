from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_make_with_trees_already_with_trees(self):
    repo = self.make_repository_with_without_trees(True)
    e = self.assertRaises(reconfigure.AlreadyWithTrees, reconfigure.Reconfigure.set_repository_trees, repo.controldir, True)
    self.assertContainsRe(str(e), "Shared repository '.*' already creates working trees.")