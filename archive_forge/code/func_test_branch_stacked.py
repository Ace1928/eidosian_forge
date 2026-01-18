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
def test_branch_stacked(self):
    trunk_tree = self.make_branch_and_tree('mainline', format='1.9')
    original_revid = trunk_tree.commit('mainline')
    self.assertRevisionInRepository('mainline', original_revid)
    out, err = self.run_bzr(['branch', '--stacked', 'mainline', 'newbranch'])
    self.assertEqual('', out)
    self.assertEqual('Created new stacked branch referring to %s.\n' % trunk_tree.branch.base, err)
    self.assertRevisionNotInRepository('newbranch', original_revid)
    new_branch = branch.Branch.open('newbranch')
    self.assertEqual(trunk_tree.branch.base, new_branch.get_stacked_on_url())