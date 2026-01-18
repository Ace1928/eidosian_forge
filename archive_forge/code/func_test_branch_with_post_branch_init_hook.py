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
def test_branch_with_post_branch_init_hook(self):
    calls = []
    branch.Branch.hooks.install_named_hook('post_branch_init', calls.append, None)
    self.assertLength(0, calls)
    self.example_branch('a')
    self.assertLength(1, calls)
    self.run_bzr('branch a b')
    self.assertLength(2, calls)