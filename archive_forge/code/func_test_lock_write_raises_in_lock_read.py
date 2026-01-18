from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_write_raises_in_lock_read(self):
    branch = self.make_branch('b')
    branch.lock_read()
    self.addCleanup(branch.unlock)
    err = self.assertRaises(errors.ReadOnlyError, branch.lock_write)