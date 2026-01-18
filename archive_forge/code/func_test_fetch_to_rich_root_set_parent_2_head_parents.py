from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_to_rich_root_set_parent_2_head_parents(self):
    self.do_test_fetch_to_rich_root_sets_parents_correctly(((ROOT_ID, b'left'), (ROOT_ID, b'right')), [(b'base', None, [('add', ('', ROOT_ID, 'directory', ''))]), (b'left', None, []), (b'right', [b'base'], []), (b'tip', [b'left', b'right'], [])])