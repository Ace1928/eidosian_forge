from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_and_unlock_leaves_repo_unlocked(self):
    branch = self.make_branch('b')
    branch.lock_write()
    branch.unlock()
    self.assertRaises(errors.LockNotHeld, branch.repository.unlock)