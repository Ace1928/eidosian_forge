from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_to_rich_root_set_parent_2_parent_2_different_id_moved(self):
    self.do_test_fetch_to_rich_root_sets_parents_correctly(((b'my-root', b'right'),), [(b'origin', None, [('add', ('', ROOT_ID, 'directory', '')), ('add', ('child', b'my-root', 'directory', ''))]), (b'base', None, []), (b'right', None, [('unversion', 'child'), ('unversion', ''), ('flush', None), ('add', ('', b'my-root', 'directory', ''))]), (b'tip', [b'base', b'right'], [('unversion', ''), ('unversion', 'child'), ('flush', None), ('add', ('', b'my-root', 'directory', ''))])], root_id=b'my-root')