from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def make_fork(self, branch):
    fork = branch.create_clone_on_transport(self.get_transport('fork'))
    self.addCleanup(fork.lock_write().unlock)
    with fork.basis_tree().preview_transform() as tt:
        tt.commit(fork, message='Commit in fork.', revision_id=b'fork-0')
    with fork.basis_tree().preview_transform() as tt:
        tt.commit(fork, message='Commit in fork.', revision_id=b'fork-1')
    return fork