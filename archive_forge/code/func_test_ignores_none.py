from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_ignores_none(self):

    def get_tag_name_1(br, revid):
        return None

    def get_tag_name_2(br, revid):
        return 'foo2'
    branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name_1, 'tagname1')
    branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name_2, 'tagname2')
    self.assertEqual('foo2', self.branch.automatic_tag_name(self.branch.last_revision()))