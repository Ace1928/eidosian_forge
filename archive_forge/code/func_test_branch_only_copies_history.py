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
def test_branch_only_copies_history(self):
    format = bzrdir.BzrDirMetaFormat1()
    format.repository_format = RepositoryFormatKnit1()
    shared_repo = self.make_repository('repo', format=format, shared=True)
    shared_repo.set_make_working_trees(True)

    def make_shared_tree(path):
        shared_repo.controldir.root_transport.mkdir(path)
        controldir.ControlDir.create_branch_convenience('repo/' + path)
        return WorkingTree.open('repo/' + path)
    tree_a = make_shared_tree('a')
    self.build_tree(['repo/a/file'])
    tree_a.add('file')
    a1 = tree_a.commit('commit a-1')
    with open('repo/a/file', 'ab') as f:
        f.write(b'more stuff\n')
    a2 = tree_a.commit('commit a-2')
    tree_b = make_shared_tree('b')
    self.build_tree(['repo/b/file'])
    tree_b.add('file')
    b1 = tree_b.commit('commit b-1')
    self.assertTrue(shared_repo.has_revision(a1))
    self.assertTrue(shared_repo.has_revision(a2))
    self.assertTrue(shared_repo.has_revision(b1))
    self.run_bzr('branch repo/b branch-b')
    pushed_tree = WorkingTree.open('branch-b')
    pushed_repo = pushed_tree.branch.repository
    self.assertFalse(pushed_repo.has_revision(a1))
    self.assertFalse(pushed_repo.has_revision(a2))
    self.assertTrue(pushed_repo.has_revision(b1))