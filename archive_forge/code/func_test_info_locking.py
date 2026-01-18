import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_locking(self):
    transport = self.get_transport()
    repo = self.make_repository('repo', shared=True, format=bzrdir.BzrDirMetaFormat1())
    repo.set_make_working_trees(False)
    repo.controldir.root_transport.mkdir('branch')
    repo_branch = controldir.ControlDir.create_branch_convenience('repo/branch', format=bzrdir.BzrDirMetaFormat1())
    transport.mkdir('tree')
    transport.mkdir('tree/checkout')
    co_branch = controldir.ControlDir.create_branch_convenience('tree/checkout', format=bzrdir.BzrDirMetaFormat1())
    co_branch.bind(repo_branch)
    transport.mkdir('tree/lightcheckout')
    lco_dir = bzrdir.BzrDirMetaFormat1().initialize('tree/lightcheckout')
    lco_dir.set_branch_reference(co_branch)
    lco_dir.create_workingtree()
    lco_tree = lco_dir.open_workingtree()
    self.assertCheckoutStatusOutput('-v tree/lightcheckout', lco_tree, repo_branch=repo_branch, verbose=True, light_checkout=True)
    with lco_tree.branch.repository.lock_write():
        self.assertCheckoutStatusOutput('-v tree/lightcheckout', lco_tree, repo_branch=repo_branch, repo_locked=True, verbose=True, light_checkout=True)
    with lco_tree.branch.lock_write():
        self.assertCheckoutStatusOutput('-v tree/lightcheckout', lco_tree, branch_locked=True, repo_locked=True, repo_branch=repo_branch, verbose=True)
    with lco_tree.lock_write():
        self.assertCheckoutStatusOutput('-v tree/lightcheckout', lco_tree, repo_branch=repo_branch, tree_locked=True, branch_locked=True, repo_locked=True, verbose=True)
    with lco_tree.lock_write(), lco_tree.branch.repository.unlock():
        self.assertCheckoutStatusOutput('-v tree/lightcheckout', lco_tree, repo_branch=repo_branch, tree_locked=True, branch_locked=True, verbose=True)
    with lco_tree.lock_write(), lco_tree.branch.unlock():
        self.assertCheckoutStatusOutput('-v tree/lightcheckout', lco_tree, repo_branch=repo_branch, tree_locked=True, verbose=True)
    with lco_tree.lock_write(), lco_tree.branch.unlock(), lco_tree.branch.repository.lock_write():
        self.assertCheckoutStatusOutput('-v tree/lightcheckout', lco_tree, repo_branch=repo_branch, tree_locked=True, repo_locked=True, verbose=True)
    with lco_tree.branch.lock_write(), lco_tree.branch.repository.unlock():
        self.assertCheckoutStatusOutput('-v tree/lightcheckout', lco_tree, repo_branch=repo_branch, branch_locked=True, verbose=True)
    if sys.platform == 'win32':
        self.knownFailure('Win32 cannot run "brz info" when the tree is locked.')