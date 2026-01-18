from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_delete_tag_invalides_cache(self):
    b, revids = self.make_write_locked_branch_with_one_tag()
    b.tags.delete_tag('one')
    self.assertEqual({}, b.tags.get_tag_dict())