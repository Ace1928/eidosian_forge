import os
from breezy import branch, controldir, errors
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import bzrdir
from breezy.bzr.knitrepo import RepositoryFormatKnit1
from breezy.tests import fixtures, test_server
from breezy.tests.blackbox import test_switch
from breezy.tests.features import HardlinkFeature
from breezy.tests.script import run_script
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.urlutils import local_path_to_url, strip_trailing_slash
from breezy.workingtree import WorkingTree
def test_branch_with_nested_trees_no_recurse(self):
    orig = self.make_branch_and_tree('source', format='development-subtree')
    subtree = self.make_branch_and_tree('source/subtree')
    self.build_tree(['source/subtree/a'])
    subtree.add(['a'])
    subtree.commit('add subtree contents')
    orig.add_reference(subtree)
    orig.commit('add subtree')
    self.run_bzr('branch --no-recurse-nested source target')
    target = WorkingTree.open('target')
    self.addCleanup(subtree.lock_read().unlock)
    basis = subtree.basis_tree()
    self.addCleanup(basis.lock_read().unlock)
    self.assertRaises(errors.NotBranchError, WorkingTree.open, 'target/subtree')