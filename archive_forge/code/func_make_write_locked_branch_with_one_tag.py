from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def make_write_locked_branch_with_one_tag(self):
    b, revids = self.make_branch_with_revision_tuple('b', 3)
    b.tags.set_tag('one', revids[0])
    self.addCleanup(b.lock_write().unlock)
    b.tags.get_tag_dict()
    return (b, revids)