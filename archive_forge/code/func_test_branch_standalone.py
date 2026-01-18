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
def test_branch_standalone(self):
    shared_repo = self.make_repository('repo', shared=True)
    self.example_branch('source')
    self.run_bzr('branch --standalone source repo/target')
    b = branch.Branch.open('repo/target')
    expected_repo_path = os.path.abspath('repo/target/.bzr/repository')
    self.assertEqual(strip_trailing_slash(b.repository.base), strip_trailing_slash(local_path_to_url(expected_repo_path)))