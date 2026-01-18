from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_to_rich_root_set_parent_1_ghost_parent(self):
    if not self.repository_format.supports_ghosts:
        raise TestNotApplicable('repository format does not support ghosts')
    self.do_test_fetch_to_rich_root_sets_parents_correctly((), [(b'tip', [b'ghost'], [('add', ('', ROOT_ID, 'directory', ''))])], allow_lefthand_ghost=True)