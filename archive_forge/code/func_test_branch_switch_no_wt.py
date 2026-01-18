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
def test_branch_switch_no_wt(self):
    self.example_branch('a')
    self.make_branch('current')
    self.run_bzr_error(['No WorkingTree exists for'], 'branch --switch ../a ../b', working_dir='current')
    a = branch.Branch.open('a')
    b = branch.Branch.open('b')
    self.assertEqual(a.last_revision(), b.last_revision())
    work = branch.Branch.open('current')
    self.assertEqual(work.last_revision(), _mod_revision.NULL_REVISION)