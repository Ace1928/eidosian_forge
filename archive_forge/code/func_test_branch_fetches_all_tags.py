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
def test_branch_fetches_all_tags(self):
    builder = self.make_branch_builder('source')
    source, rev1, rev2 = fixtures.build_branch_with_non_ancestral_rev(builder)
    source.tags.set_tag('tag-a', rev2)
    source.get_config_stack().set('branch.fetch_tags', True)
    self.run_bzr('branch source new-branch')
    new_branch = branch.Branch.open('new-branch')
    self.assertEqual(rev2, new_branch.tags.lookup_tag('tag-a'))
    new_branch.repository.get_revision(rev2)