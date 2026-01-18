from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def make_unsynced_branch_reconfiguration(self):
    parent = self.make_branch_and_tree('parent')
    parent.commit('commit 1')
    child = parent.controldir.sprout('child').open_workingtree()
    child.commit('commit 2')
    return reconfigure.Reconfigure.to_lightweight_checkout(child.controldir)