from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_merge_empty_tags(self):
    b1 = self.make_branch('b1')
    b2 = self.make_branch('b2')
    b1.tags.merge_to(b2.tags)